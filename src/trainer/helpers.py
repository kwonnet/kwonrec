import numpy as np
import pandas as pd
import tensorflow as tf

def run_model_similarity_score(model, users: list[str]):
    print("\n--- Initiated Model Similarity Score ---", flush=True)
    # Use the first two different users
    if len(users) < 2:
        print("Not enough users for similarity test.")
        return

    # Get their embeddings using the full model (which calls the user_tower)
    # We call the model directly to ensure fusion logic is applied
    emb1 = model.user_model(tf.constant([users[0]]))
    emb2 = model.user_model(tf.constant([users[1]]))

    # Check the distance between them
    cosine_sim = tf.reduce_sum(tf.multiply(emb1, emb2))
    print(f"Similarity Score: {cosine_sim.numpy()}", flush=True)


def run_scann_search_test(scann, user_id):
    """
    Initiate scann search so that the saved version can work when loaded 
    """
    print("\n--- Initiated Scann Search Test ---", flush=True)
        
    scores, post_ids = scann(tf.constant([user_id]), k=1)

    score = scores.numpy()[0][0]
    post_id = post_ids.numpy()[0][0].decode('utf-8')

    print(f"Top recommendation - {post_id} - score {score}", flush=True)


def run_diversity_test(scann, user_ids, k=10):
    print("\n--- Running Diversity Test ---", flush=True)
    all_recs = {}
    
    for uid in user_ids:
        _, post_ids = scann(tf.constant([uid]), k=k)
        
        all_recs[uid] = set([pid.decode('utf-8') if isinstance(pid, bytes) else str(pid) 
                             for pid in post_ids[0].numpy()])

    u_list = list(all_recs.keys())
    overlaps = []
    
    for i in range(len(u_list)):
        for j in range(i + 1, len(u_list)):
            u1, u2 = u_list[i], u_list[j]
            intersection = all_recs[u1].intersection(all_recs[u2])
            union = all_recs[u1].union(all_recs[u2])
            jaccard = len(intersection) / len(union)
            overlaps.append(jaccard)
            print(f"Overlap User {i} vs {j}: {jaccard:.2%} ({len(intersection)} shared posts)", flush=True)

    avg_overlap = np.mean(overlaps) if overlaps else 0
    print(f"\nAverage Cross-User Overlap: {avg_overlap:.2%}", flush=True)
    
    if avg_overlap > 0.50:
        print("⚠️ WARNING: High Overlap. Popularity bias detected.", flush=True)
    elif avg_overlap < 0.15:
        print("✅ SUCCESS: High Diversity.", flush=True)
    else:
        print("ℹ️ MODERATE: Balanced personalization.", flush=True)


def run_cold_start_test(scann, user_id):
    print("\n--- Running Cold Start Test ---", flush=True)    
    # 1. New User ID + Context
    new_user_query = tf.constant(["brand_new_user_999"])
    existing_user_query = tf.constant([user_id])
    
    # 2. Get recommendations
    scores_new, ids_new = scann(new_user_query, k=10)
    scores_old, ids_old = scann(existing_user_query, k=10)
    
    # 3. Format results
    new_recs = [pid.decode('utf-8') if isinstance(pid, bytes) else str(pid) for pid in ids_new[0].numpy()]
    old_recs = [pid.decode('utf-8') if isinstance(pid, bytes) else str(pid) for pid in ids_old[0].numpy()]
    
    # 4. Analyze Overlap
    intersection = set(new_recs).intersection(set(old_recs))
    
    print(f"Known User ({user_id}) Top 3: {old_recs[:3]}", flush=True)
    print(f"New User (Cold Start) Top 3: {new_recs[:3]}", flush=True)
    print(f"\nOverlap between Known and New User: {len(intersection)} posts", flush=True)
    
    if len(intersection) > 7:
        print("⚠️ RESULT: High Overlap.", flush=True)
    else:
        print("✅ RESULT: Personalization working.", flush=True)
