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

def prepare_datasets(df: pd.DataFrame):
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
    
    # === CRITICAL FIX: Handle positive and negative weights separately ===
    # Positive weights: prevent very old likes/bookmarks from vanishing completely
    positive_mask = df['weight'] > 0
    df.loc[positive_mask, 'weight'] = df.loc[positive_mask, 'weight'].clip(lower=0.01)
    
    # Negative weights (dislikes, reports): 
    # - Keep them negative
    # - Optional: amplify slightly so negatives have strong repulsion
    # - Optional: floor at a reasonable lower bound (e.g., -10) to avoid extremes
    negative_mask = df['weight'] < 0
    df.loc[negative_mask, 'weight'] = df.loc[negative_mask, 'weight'] * 1.2  # Optional: boost negative signal
    df.loc[negative_mask, 'weight'] = df.loc[negative_mask, 'weight'].clip(upper=-0.5)  # Ensure at least moderate repulsion

    # Zero weights remain zero (neutral impressions that decayed fully)

    # Current age = how old the post is RIGHT NOW (for recency at inference)
    df['current_age_hours'] = (now - df['post_created_at']).dt.total_seconds() / 3600
    df['current_age_hours'] = np.log1p(df['current_age_hours'].clip(lower=0))

    return df


def stratified_dataset_split(df: pd.DataFrame, test_ratio: float = 0.2, val_ratio: float = 0.1):
    df = prepare_datasets(df)

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

