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

def fetch_user_interactions(user_id: str, limit: int = 50) -> pd.DataFrame:
    client = get_clickhouse_client()

    query = """
    SELECT 
        user_id, post_id, author_id, type, weight, label, timestamp, embedding,
        post_created_at, post_age_hours, post_created_hour, post_day_of_week
    FROM interactions 
    WHERE user_id = %(user_id)s
    ORDER BY timestamp DESC
    LIMIT %(limit)d
    """
    # params = {'last_ts': int(last_timestamp.timestamp() * 1000)}  # Milliseconds for DateTime64(3)
    rows = client.execute(query, params={"user_id": user_id, "limit": limit}, columnar=True)
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