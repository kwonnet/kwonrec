
from .config import  MODEL_SERVING_PATH, MAPPINGS_PATH, SCANN_INDEX_PATH
import tensorflow as tf
import tensorflow_recommenders as tfrs
from datetime import datetime
import numpy as np


def get_recommendations(user_id: str, limit: int):
    print(f"Recommend user - {user_id} with limit {limit}")
    
    # Load served model and index
    # Note: In a production environment, you should load this *once* outside the function
    # or use a global cache to avoid constant disk I/O.
    scann = tf.saved_model.load(SCANN_INDEX_PATH)

    print(f"saved serving_index")

    print(scann.signatures.keys())

    print(scann)

    print(f"printed serving_index")
    
    # Convert the user_id to the expected TensorFlow constant/tensor format

    now = datetime.now()
    scores_raw, post_ids_raw = scann({
        "current_hour": tf.constant([now.hour], dtype=tf.int32),
        "current_day": tf.constant([now.weekday()], dtype=tf.int32),
        "user_id": tf.constant([user_id])
    })

    # scores_raw, post_ids_raw = scann(tf.constant([user_id]))
    
    # Flatten the tensor output to standard Python lists
    post_ids = post_ids_raw.numpy().flatten().tolist()[:limit]
    scores = scores_raw.numpy().flatten().tolist()[:limit]
    
    # --- END LIMIT FIX ---
    
    recommendations = [
        {"id": post_id, "score": score}
        for post_id, score in zip(post_ids, scores)
    ]

    return recommendations
