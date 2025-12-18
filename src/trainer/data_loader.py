# data_fetch.py
import numpy as np
import pandas as pd
import tensorflow as tf
from clickhouse_driver import Client
from datetime import datetime
from ..config import (
    CLICKHOUSE_HOST, CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD, CLICKHOUSE_DB, BATCH_SIZE
)
from typing import Tuple
import math # <-- Import math for ceiling function
from sklearn.model_selection import train_test_split

def show_dataset_len(dataset):
    # Assuming interactions_ds is your dataset
    cardinality = dataset.cardinality().numpy()

    if cardinality == tf.data.INFINITE_CARDINALITY:
        print("The dataset is infinite.",flush=True)
    elif cardinality == tf.data.UNKNOWN_CARDINALITY:
        print("The dataset size is unknown (e.g., from a generator).",flush=True)
    else:
        # This will return the number of elements (interactions) in the dataset
        print(f"Total number of elements (interactions): {cardinality}",flush=True)


def get_clickhouse_client() -> Client:
    try:
        return Client(
            host=CLICKHOUSE_HOST,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB,
            secure=True,
        )
    except Exception as e:
        print(f"Error connecting to ClickHouse: {e}")
        raise e

# Step 1: Retrieve interaction data from ClickHouse
def fetch_new_interactions() -> pd.DataFrame:
    client = get_clickhouse_client()

    query = """
    SELECT user_id, post_id, author_id, type, weight, label, timestamp, embedding, post_created_at, post_age_hours, post_created_hour, post_day_of_week
    FROM interactions ORDER BY created_at ASC 
    """
    # params = {'last_ts': int(last_timestamp.timestamp() * 1000)}  # Milliseconds for DateTime64(3)
    rows = client.execute(query, columnar=True)
    if not rows:
        return pd.DataFrame()

    # num_records = len(rows[0])
    print(f"Total columns fetched: {len(rows)} (12 expected)", flush=True)
    # print(f"Total records (rows) fetched: {num_records}", flush=True)

    columns = ['user_id', 'post_id', 'author_id', 'type', 'weight', 'label', 'timestamp', 'embedding',
               'post_created_at', 'post_age_hours', 'post_created_hour', 'post_day_of_week']
    df = pd.DataFrame({col: list(vals) for col, vals in zip(columns, rows)})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['post_created_at'] = pd.to_datetime(df['post_created_at'])
    df['embedding'] = df['embedding'].apply(np.array)  # Convert lists to numpy arrays
    return df


def convert_df_to_tf_dataset(df: pd.DataFrame):
    result = tf.data.Dataset.from_tensor_slices({
        'user_id': df['user_id'].values.astype(str),          # tf.string
        'post_id': df['post_id'].values.astype(str),          # tf.string
        'author_id': df['author_id'].values.astype(str),      # tf.string (if used elsewhere)

        'label': df['label'].values.astype(np.float32),       # tf.float32
        'weight': df['weight'].values.astype(np.float32),     # tf.float32 (now decayed)

        # Post content embedding (384-dim)
        'embedding': np.array(df['embedding'].tolist(), dtype=np.float32),

        # Historical age: how old the post was WHEN the user interacted
        'post_age_hours_interaction': df['post_age_hours_interaction'].values.astype(np.float32),

        # Current age: how old the post is RIGHT NOW (also used in training for consistency)
        'current_age_hours': df['current_age_hours'].values.astype(np.float32),

        # Static post timing features
        'post_created_hour': df['post_created_hour'].values.astype(np.int32),
        'post_day_of_week': df['post_day_of_week'].values.astype(np.int32),
    })
    return result

def convert_df_to_tf_candidate_ds(df: pd.DataFrame):
    # Use only unique posts to avoid duplicates in the candidate set
    candidate_df = df.drop_duplicates(subset=['post_id']).copy()

    candidates_ds = tf.data.Dataset.from_tensor_slices({
        'post_id': candidate_df['post_id'].values.astype(str),

        'author_id': candidate_df['author_id'].values.astype(str),
        # Post content embedding
        'embedding': np.array(candidate_df['embedding'].tolist(), dtype=np.float32),

        # Current age at the time of indexing/serving → critical for recency bias
        'current_age_hours': candidate_df['current_age_hours'].values.astype(np.float32),

        # Static features
        'post_created_hour': candidate_df['post_created_hour'].values.astype(np.int32),
        'post_day_of_week': candidate_df['post_day_of_week'].values.astype(np.int32),

        # Optional: zero-fill historical age since it's not used at inference
        # This allows the same get_candidate_embedding logic to work without errors
        'post_age_hours_interaction': np.zeros(len(candidate_df), dtype=np.float32),
    })
    return candidates_ds


def prepare_stratified_datasets(df: pd.DataFrame, test_ratio: float = 0.2, val_ratio: float = 0.1):
    # 1. Force all date columns to be Timezone-Aware (UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['post_created_at'] = pd.to_datetime(df['post_created_at'], utc=True)
    
    # 2. Establish "Now" as a UTC-aware anchor (used for decay and current age)
    now = pd.Timestamp.now(tz='UTC')

    # 3. Calculate historical post age AT THE TIME OF INTERACTION (for training)
    df['post_age_hours_interaction'] = (df['timestamp'] - df['post_created_at']).dt.total_seconds() / 3600
    df['post_age_hours_interaction'] = np.log1p(df['post_age_hours_interaction'].clip(lower=0))

    # 4. Static post features (same for training and inference)
    df['post_created_hour'] = df['post_created_at'].dt.hour.astype('uint8')
    df['post_day_of_week'] = df['post_created_at'].dt.dayofweek.astype('uint8')

    # 5. APPLY TEMPORAL DECAY TO INTERACTION WEIGHTS → Focus on recent user behavior
    df['days_old_from_now'] = (now - df['timestamp']).dt.total_seconds() / 86400
    decay_lambda = 0.05  # Tune this: 0.03 (slow decay) to 0.1 (strong recency)
    df['weight'] = df['weight'] * np.exp(-decay_lambda * df['days_old_from_now'])
    
    # === STRONGER NEGATIVE HANDLING ===
    positive_mask = df['weight'] > 0
    negative_mask = df['weight'] < 0

    # Positives: gentle floor to avoid vanishing
    df.loc[positive_mask, 'weight'] = df.loc[positive_mask, 'weight'].clip(lower=0.01)

    # Negatives: amplify + preserve full strength
    df.loc[negative_mask, 'weight'] = df.loc[negative_mask, 'weight'] * 2.0   # or 3.0 for very strong repulsion
    
    # Current age = how old the post is RIGHT NOW (for recency at inference)
    df['current_age_hours'] = (now - df['post_created_at']).dt.total_seconds() / 3600
    df['current_age_hours'] = np.log1p(df['current_age_hours'].clip(lower=0))

    # 6. Stratified Split (by user)
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df['user_id'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio / (1 - test_ratio),
        stratify=train_val_df['user_id'], random_state=42
    )

    # 7. Convert to TF Datasets
    train_ds = convert_df_to_tf_dataset(train_df)
    val_ds = convert_df_to_tf_dataset(val_df)
    test_ds = convert_df_to_tf_dataset(test_df)

    # Candidate datasets
    train_candidates_ds = convert_df_to_tf_candidate_ds(train_df)   
    all_candidates_ds = convert_df_to_tf_candidate_ds(df)

    # 8. Vocabularies
    unique_user_ids = np.unique(df['user_id'].values)
    unique_post_ids = np.unique(df['post_id'].values)
    unique_author_ids = np.unique(df['author_id'].values)

    return (
        train_ds, val_ds, test_ds,
        train_candidates_ds, all_candidates_ds,
        unique_user_ids, unique_post_ids, unique_author_ids
    )


def stratified_dataset_split(
    df: pd.DataFrame, 
    test_ratio: float = 0.2, 
    val_ratio: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Splits the dataframe into training, validation, and testing sets 
    using a Stratified Temporal strategy (Warm Start).
    
    For each user, interactions are sorted by time.
    - The last `test_ratio` % go to Test.
    - The `val_ratio` % before that go to Validation.
    - The rest go to Training.

    Args:
        df (pd.DataFrame): The source DataFrame containing 'user_id' and 'timestamp'.
        test_ratio (float): The proportion of recent interactions per user for the test set.
        val_ratio (float): The proportion of interactions per user for the validation set.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """

    # 1. Sort Data to ensure Temporal Order
    # If no timestamp exists, we shuffle to create a random stratified split
    if 'timestamp' in df.columns:
        df = df.sort_values(by=['user_id', 'timestamp'])
    else:
        print("WARNING: No 'timestamp' column found. Performing random stratified split.")
        df = df.sample(frac=1, random_state=42).sort_values(by='user_id')

    # 2. Calculate Rank Percentile per User
    # This assigns a score from 0.0 to 1.0 for each interaction within the user's history
    df['rank_pct'] = df.groupby('user_id').cumcount() / df.groupby('user_id')['user_id'].transform('count')

    # 3. Define Split Cutoffs
    # Example: Test=0.2, Val=0.1
    # Train: 0.0 to 0.7
    # Val:   0.7 to 0.8
    # Test:  0.8 to 1.0
    test_cutoff = 1.0 - test_ratio
    val_cutoff = test_cutoff - val_ratio

    # 4. Apply the split
    train_df = df[df['rank_pct'] <= val_cutoff]
    val_df = df[(df['rank_pct'] > val_cutoff) & (df['rank_pct'] <= test_cutoff)]
    test_df = df[df['rank_pct'] > test_cutoff]

    # Clean up helper column
    train_df = train_df.drop(columns=['rank_pct'])
    val_df = val_df.drop(columns=['rank_pct'])
    test_df = test_df.drop(columns=['rank_pct'])

    # 5. Print summary (Matching your preferred style)
    total_rows = len(df)
    print(f"Total Interactions: {total_rows}", flush=True)
    
    print(f"Interactions in Test Set: {len(test_df)} ({len(test_df)/total_rows:.1%})", flush=True)
    print(f"Interactions in Validation Set: {len(val_df)} ({len(val_df)/total_rows:.1%})", flush=True)
    print(f"Interactions in Training Set: {len(train_df)} ({len(train_df)/total_rows:.1%})", flush=True)

    # 6. Convert to TensorFlow Datasets
    # We use dict(dataframe) to correctly map columns to feature names
    train_ds = convert_df_to_tf_dataset(train_df) 
    val_ds = convert_df_to_tf_dataset(val_df)
    test_ds = convert_df_to_tf_dataset(test_df)

    # df to candidate dataset (Post-level features)
    
    all_candidates_ds = convert_df_to_tf_candidate_ds(df)
    
    train_candidates_ds = convert_df_to_tf_candidate_ds(train_df)

    # Unique users and posts for vocabularies
    unique_user_ids = np.unique(df['user_id'].values)
    unique_post_ids = np.unique(df['post_id'].values)

    # Return all datasets and uniques
    return train_ds, val_ds, test_ds, train_candidates_ds, all_candidates_ds, unique_user_ids, unique_post_ids



def user_based__dataset_split(
    unique_user_ids: np.ndarray, 
    interactions_ds: tf.data.Dataset, 
    test_ratio: float = 0.2, 
    val_ratio: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Splits the interactions dataset into training, validation, and testing sets 
    based on user IDs, ensuring all three sets are mutually exclusive by user.
    Uses math.ceil for non-zero minimum splits when the user count is small.

    Args:
        # ... (arguments remain the same)
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    
    # 1. Shuffle unique user IDs IN-PLACE
    total_users = len(unique_user_ids)
    
    # CRITICAL CHECK: Ensure we have enough users to split
    if total_users < 3 and (test_ratio > 0 or val_ratio > 0):
        print(f"WARNING: Total users ({total_users}) is too small to perform a three-way split. Returning all users in training set.")
        return interactions_ds, interactions_ds.take(0), interactions_ds.take(0)

    np.random.shuffle(unique_user_ids) 
    
    # 2. Define split sizes using math.ceil
    print(f"Total Unique Users: {total_users}", flush=True)

    # Calculate Test User Size - Use ceil to ensure at least 1 user if ratio > 0
    test_user_size = math.ceil(test_ratio * total_users)
    # Ensure test size is not larger than total users
    test_user_size = min(test_user_size, total_users) 
    print(f"Calculated Test Users: {test_user_size}", flush=True)

    # Calculate Validation User Size (ratio is applied to the REMAINING users)
    remaining_users = total_users - test_user_size
    val_user_size = math.ceil(val_ratio * remaining_users)
    # Ensure val size is not larger than the remaining users
    val_user_size = min(val_user_size, remaining_users)
    print(f"Calculated Validation Users: {val_user_size}", flush=True)
    
    # 3. Split the users array
    # Test users are the first part
    test_user_ids = unique_user_ids[:test_user_size]
    
    # Validation users are the next part
    val_user_ids = unique_user_ids[test_user_size : test_user_size + val_user_size]
    
    # Training users are the remainder
    train_user_ids = unique_user_ids[test_user_size + val_user_size :]
    
    # Print the correct calculated size for the remaining training set
    print(f"Calculated Training Users: {len(train_user_ids)}", flush=True) 
    
    # Convert to TensorFlow tensors for efficient filtering
    TEST_USERS_TENSOR = tf.constant(test_user_ids)
    VAL_USERS_TENSOR = tf.constant(val_user_ids)
    TRAIN_USERS_TENSOR = tf.constant(train_user_ids)

    # Print summary (helpful for debugging)
    print(f"Users in Test Set: {len(test_user_ids)} ({len(test_user_ids)/total_users:.1%})", flush=True)
    print(f"Users in Validation Set: {len(val_user_ids)} ({len(val_user_ids)/total_users:.1%})", flush=True)
    print(f"Users in Training Set: {len(train_user_ids)} ({len(train_user_ids)/total_users:.1%})", flush=True)


    # 4. Filtering function (no change needed here)
    def is_in_set(user_id, user_set):
        return tf.reduce_any(tf.equal(user_id, user_set))

    # 5. Apply the split to interactions_ds (no change needed here)
    test_ds = interactions_ds.filter(lambda x: is_in_set(x['user_id'], TEST_USERS_TENSOR))
    val_ds = interactions_ds.filter(lambda x: is_in_set(x['user_id'], VAL_USERS_TENSOR))
    train_ds = interactions_ds.filter(lambda x: is_in_set(x['user_id'], TRAIN_USERS_TENSOR))

    return train_ds, val_ds, test_ds

