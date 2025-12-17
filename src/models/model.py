import os
import datetime
import tensorflow as tf
import tensorflow_recommenders as tfrs
from ..config import EMBEDDING_DIM

class HybridRecommenderModel(tfrs.models.Model):
    def __init__(self, 
                 rating_weight: float,
                 retrieval_weight: float, 
                 unique_user_ids, 
                 unique_post_ids, 
                 candidates_ds, 
                 embedding_dim=EMBEDDING_DIM):
        super().__init__()
        
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

        # --- 1. DEFINE INDEPENDENT LAYERS ---
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim),
        ])
        self.user_embedding_layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim),
        ])

        self.user_context_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(embedding_dim, activation="relu") 
        ])

        self.user_tower_fusion = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim) 
        ])

        # --- 2. BUILD FORMAL QUERY MODEL (Functional API) ---
        # This solves the "Weights not created" and "Untracked resource" errors
        u_id_in = tf.keras.Input(shape=(), dtype=tf.string, name="user_id")
        u_hr_in = tf.keras.Input(shape=(), dtype=tf.int32, name="current_hour")
        u_day_in = tf.keras.Input(shape=(), dtype=tf.int32, name="current_day")

        u_id_emb = self.user_embedding_layer(u_id_in)
        u_ctx_concat = tf.concat([
            tf.one_hot(u_hr_in, 24),
            tf.one_hot(u_day_in, 7)
        ], axis=1)
        u_ctx_emb = self.user_context_model(u_ctx_concat)
        
        u_final_emb = self.user_tower_fusion(tf.concat([u_id_emb, u_ctx_emb], axis=1))

        self.query_model = tf.keras.Model(
            inputs={"user_id": u_id_in, "current_hour": u_hr_in, "current_day": u_day_in},
            outputs=u_final_emb,
            name="query_model"
        )

        # --- 3. POST TOWER COMPONENTS ---
        self.post_id_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_post_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_post_ids) + 1, embedding_dim),
            tf.keras.layers.Dense(embedding_dim)
        ])
        
        self.post_content_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
        ])
        
        self.post_context_model = tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
        ])
        
        self.post_tower_fusion = tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        # --- 4. RANKING & TASKS ---
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.map(lambda x: (x['post_id'], self.get_candidate_embedding(x)))
            )
        )

        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def get_candidate_embedding(self, features):
        """Logic for the Post Tower"""
        p_id_emb = self.post_id_model(features['post_id'])
        p_cont_emb = self.post_content_model(features['embedding'])
        # Inside get_candidate_embedding
        age_val = tf.cast(features['post_age_hours'], tf.float32)
        # Use log(x + 1) to compress the range
        log_age = tf.math.log1p(tf.expand_dims(age_val, axis=-1))
        p_feats = self.post_context_model([
            log_age,
            tf.one_hot(tf.cast(features['post_created_hour'], tf.int32), 24),
            tf.one_hot(tf.cast(features['post_day_of_week'], tf.int32), 7)
        ])
        
        return self.post_tower_fusion([p_id_emb, p_cont_emb, p_feats])

    def call(self, features: dict[str, tf.Tensor]):
        # Slice the dictionary to only include what the query_model needs
        # This removes the "UserWarning: Input dict contained keys... which did not match"
        query_features = {
            "user_id": features["user_id"],
            "current_hour": features["current_hour"],
            "current_day": features["current_day"]
        }
        
        # Use the sliced dict for the query side
        user_embeddings = self.user_model(features["user_id"])
        
        # Post side still needs the full 'features' dict for embeddings, age, etc.
        post_embeddings = self.get_candidate_embedding(features)
        
        rating_predictions = self.rating_model(
            tf.concat([user_embeddings, post_embeddings], axis=1)
        )
        
        return user_embeddings, post_embeddings, rating_predictions
        
    
    def compute_loss(self, features, training=False):
        labels = features["weight"]
        
        user_embeddings, post_embeddings, rate_predictions = self(features)
        
        rate_loss = self.rating_task(labels=labels, predictions=rate_predictions)
        retrieval_loss = self.retrieval_task(user_embeddings, post_embeddings)
        
        return self.rating_weight * rate_loss + self.retrieval_weight * retrieval_loss
