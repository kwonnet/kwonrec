# train.py
import pickle
import redis
from datetime import datetime
from ..config import (LEARNING_RATE, NUM_EPOCHS,  MAPPINGS_PATH, SCANN_INDEX_PATH, MODEL_SERVING_PATH, MODEL_WEIGHTS_PATH, RATING_WEIGHT, RETRIEVAL_WEIGHT, BATCH_SIZE, LOG_DIR
)
from .data_loader import fetch_new_interactions, prepare_stratified_datasets, user_based__dataset_split, stratified_dataset_split
from .helpers import  run_diversity_test, run_cold_start_test, run_scann_search_test, run_model_similarity_score
from ..models.model import HybridRecommenderModel
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os

# Step 2 & 3: Load or initialize model, train on new data, save
def train_full_pipeline():
    
    df = fetch_new_interactions()
    if df.empty:
        print("No new interactions. Skipping training.")
        return
    # stratified dataset split
    train_ds, val_ds, test_ds, train_candidates_ds, all_candidates_ds, unique_user_ids, unique_post_ids, unique_author_ids = prepare_stratified_datasets(df)

    # stratified dataset split
    # train_ds, val_ds, test_ds, train_candidates_ds, all_candidates_ds, unique_user_ids, unique_post_ids = stratified_dataset_split(df)

    # 2. Caching and Batching (Looks good)
    cached_train =  train_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    cached_val =  val_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    cached_test =  test_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    # candidate ds
    indexing_candidates_ds =  train_candidates_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Initialize or load model
    model = HybridRecommenderModel(
        rating_weight=RATING_WEIGHT,
        retrieval_weight=RETRIEVAL_WEIGHT,
        unique_user_ids=unique_user_ids, 
        unique_post_ids=unique_post_ids,
        unique_author_ids=unique_author_ids,
        candidates_ds=indexing_candidates_ds)

    # if os.path.exists(MODEL_WEIGHTS_PATH):
    #     model.load_weights(MODEL_WEIGHTS_PATH)
    #     print("Loaded previous model weights for incremental training.")
    
    # # Compile
    # learning_rate = LEARNING_RATE if not os.path.exists(MODEL_WEIGHTS_PATH) else LEARNING_RATE * 0.5
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))

    # early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        # monitor='val_factorized_top_k/top_100_categorical_accuracy',
        monitor='val_factorized_top_k/top_50_categorical_accuracy',
        # monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Set up TensorBoard callback
    log_dir = f"{LOG_DIR}/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Unique dir per run
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log histograms every epoch (for weights/activations)
        embeddings_freq=1,  # If you log embeddings (see optional below)
        update_freq='epoch'  # Log at epoch end for efficiency
    )

    # saving weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_WEIGHTS_PATH,
        monitor='val_factorized_top_k/top_50_categorical_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    print("About to train HybridRecommenderModel", flush=True)
    # Train incrementally
    model.fit(
        cached_train, 
        validation_data=cached_val, 
        epochs=NUM_EPOCHS, 
        verbose=1,
        callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback]
    )

    print("About to evaluate HybridRecommenderModel", flush=True)
    # evaluate
    # Evaluate the model on the test set
    test_results = model.evaluate(cached_test, return_dict=True, verbose=1)
    print("\nTest Evaluation Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save weights for future incremental training
    # model.save_weights(MODEL_WEIGHTS_PATH)
    
    # Build index for serving (ScaNN for efficiency)
    # Build the (post_id, embedding) dataset for ScaNN
    candidate_embeddings_ds = all_candidates_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(
        # model.get_candidate_embedding
        lambda x: (
            x['post_id'],
            model.get_candidate_embedding(x)  # ← Uses your existing method!
        )
    )
   
    scann = tfrs.layers.factorized_top_k.ScaNN(
        query_model=model.user_model, 
        k=100,                          # Return the top 100 items
        num_leaves=100,
        num_leaves_to_search=10,            
        num_reordering_candidates=400,  # Search 400, then pick the best 100
        name="recommender"
    )
    scann.index_from_dataset(candidate_embeddings_ds)

    # Initiate similarity score
    run_model_similarity_score(model, unique_user_ids.tolist())

    # Performing scann search to make the saved scann work when loaded
    run_scann_search_test(scann,unique_user_ids[0])

    # Run Validation Tests (using the fixed testing utils)
    run_diversity_test(scann, unique_user_ids[:5].tolist())
    
    # Perform cold start test
    run_cold_start_test(scann, unique_user_ids[0])


    # Save model and index
    # tf.saved_model.save(model, MODEL_SERVING_PATH)
    # tf.keras.models.save_model(
    #     model,
    #     MODEL_SERVING_PATH,
    #     overwrite=True,
    #     include_optimizer=True,
    #     save_format=None,
    #     signatures=None,
    #     options=None
    # )

    print("Saving scann index for inference.........", flush=True)

    tf.saved_model.save(scann, SCANN_INDEX_PATH, options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))
    
    print("Model trained and saved successfully.", flush=True)

if __name__ == "__main__":
    train_full_pipeline()
