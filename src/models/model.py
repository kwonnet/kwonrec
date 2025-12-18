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
                 unique_author_ids,
                 candidates_ds, 
                 embedding_dim=EMBEDDING_DIM):
        super().__init__()
        
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

        # User tower
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim),
        ])

        # Post ID tower
        self.post_id_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_post_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_post_ids) + 1, embedding_dim),
        ])

        self.author_id_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_author_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_author_ids) + 1, embedding_dim),
        ])
        
        # Post content (embedding) tower
        self.post_content_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
        ])
        
        # Post contextual features tower — now includes both historical and current age
        self.post_context_model = tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
        ])
        
        # Final post tower fusion
        self.post_tower_fusion = tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        # Ranking head
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        # Retrieval task
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
        """Compute post embedding — used both in training and candidate indexing"""
        p_id_emb = self.post_id_model(features['post_id'])
        p_cont_emb = self.post_content_model(features['embedding'])

        p_author_emb = self.author_id_model(features['author_id'])
        
        # Historical age at interaction time (only meaningful in training data)
        age_interaction = tf.expand_dims(
            tf.cast(features.get('post_age_hours_interaction', 0.0), tf.float32), -1
        )
        age_interaction = tf.math.log1p(tf.maximum(age_interaction, 0.0))
        
        # Current age — always available, critical for recency bias at serving
        current_age = tf.expand_dims(tf.cast(features['current_age_hours'], tf.float32), -1)
        current_age = tf.math.log1p(tf.maximum(current_age, 0.0))
        
        # One-hot temporal features
        hour_onehot = tf.one_hot(tf.cast(features['post_created_hour'], tf.int32), 24)
        dow_onehot = tf.one_hot(tf.cast(features['post_day_of_week'], tf.int32), 7)
        
        # Concatenate all context
        context_emb = self.post_context_model([
            age_interaction,      # Helps model historical freshness patterns
            current_age,          # Drives recency boost at inference
            hour_onehot,
            dow_onehot
        ])
        
        return self.post_tower_fusion([p_id_emb, p_cont_emb, p_author_emb, context_emb])

    def call(self, features: dict[str, tf.Tensor]):
        user_embeddings = self.user_model(features["user_id"])
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