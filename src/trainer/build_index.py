# build_index.py
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pickle
import numpy as np
from ..models.model import SocialRecModel
from .data_loader import fetch_candidates
from ..config import MODEL_PATH, MAPPINGS_PATH, SCANN_INDEX_PATH

def build_scann_index():
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"SocialRecModel": SocialRecModel})
    
    post_ids, author_ids, post_embeddings = fetch_candidates()
    
    dataset = tf.data.Dataset.from_tensor_slices({
        "post_id": post_ids,
        "author_id": author_ids,
        "post_embedding": post_embeddings
    })
    
    post_embeddings_ds = dataset.batch(512).map(model.post_tower)
    post_ids_ds = dataset.batch(512).map(lambda x: x["post_id"])
    
    scann_layer = tfrs.layers.factorized_top_k.ScaNN(
        model.user_model,
        num_leaves=1000,
        num_reordering_candidates=1000,
        num_leaves_to_search=100,
    )
    scann_layer.index_from_dataset(
        tf.data.Dataset.zip((post_ids_ds, post_embeddings_ds))
    )
    
    # Save index separately
    tf.saved_model.save(scann_layer, SCANN_INDEX_PATH)
    
    # Update mappings
    with open(MAPPINGS_PATH, "wb") as f:
        pickle.dump({
            "post_ids": post_ids,
            "author_ids": author_ids,
            "post_embeddings": post_embeddings  # optional, only if needed elsewhere
        }, f)
    
    print("ScaNN index built and saved!")

if __name__ == "__main__":
    build_scann_index()