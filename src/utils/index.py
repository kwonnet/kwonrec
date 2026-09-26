import redis
import tensorflow as tf
import pandas as pd
import numpy as np

from ..config import (
    REDIS_HOST, REDIS_PORT, USER_EMB_PREFIX, USER_MODEL_PATH
)
from datetime import datetime
from .db import fetch_user_interactions

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)  # use your Redis service name

EMB_TTL = 86400  # 24 hours — or shorter if you want more frequent refresh

# Global or cached load
user_tower = None

def get_user_tower():
    global user_tower
    if user_tower is None:
        user_tower = tf.saved_model.load(USER_MODEL_PATH)
    return user_tower

def refresh_user_embedding(user_id: str):
    user_model = get_user_tower()
    
    # Fetch recent history
    history_df = fetch_user_interactions(user_id, limit=50)  # last 50 interactions
    
    if history_df.empty:
        # Fallback to single ID
        user_input = tf.constant([[user_id]])
        emb = user_model(user_input)
    else:
        ds = convert_df_to_tf_dataset(prepare_datasets(history_df)).batch(32)
        embs = [user_model(batch["user_id"]) for batch in ds]
        emb = tf.reduce_mean(tf.concat(embs, axis=0), axis=0)
    
    # Cache
    serialized = tf.io.serialize_tensor(emb).numpy()
    redis_client.set(f"{USER_EMB_PREFIX}{user_id}", serialized, ex=EMB_TTL)

    print(f"Current user embeddings saved successfully - {user_id}", flush=True)



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