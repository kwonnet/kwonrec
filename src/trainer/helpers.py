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
    print("--- Initiated Scann Search Test ---")
    
    # Use default number of neighbors (no 'k' argument!)
    scores, post_ids = scann(tf.constant([user_id]))
    
    top_post_id = post_ids.numpy().flatten()[0]
    if isinstance(top_post_id, bytes):
        top_post_id = top_post_id.decode('utf-8')
    
    top_score = float(scores.numpy().flatten()[0])
    
    print(f"Top recommendation - {top_post_id} - score {top_score}")


def run_diversity_test(scann, user_ids, limit=10):
    print("\n--- Running Diversity Test ---", flush=True)
    all_recs = {}
    
    for uid in user_ids:
        # Use default k — no 'k' argument
        scores, post_ids = scann(tf.constant([uid]))
        
        # Take only up to 'limit' results
        post_ids_flat = post_ids.numpy().flatten()[:limit]
        
        post_ids_set = set()
        for pid in post_ids_flat:
            if isinstance(pid, bytes):
                post_ids_set.add(pid.decode('utf-8'))
            else:
                post_ids_set.add(str(pid))
        
        all_recs[uid] = post_ids_set

    u_list = list(all_recs.keys())
    overlaps = []
    
    for i in range(len(u_list)):
        for j in range(i + 1, len(u_list)):
            u1, u2 = u_list[i], u_list[j]
            intersection = all_recs[u1].intersection(all_recs[u2])
            union = all_recs[u1].union(all_recs[u2])
            jaccard = len(intersection) / len(union) if union else 0
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


def run_cold_start_test(scann, user_id, limit=10):
    print("\n--- Running Cold Start Test ---", flush=True)    
    
    # 1. Existing user
    scores_old, ids_old = scann(tf.constant([user_id]))
    
    # 2. Cold start user (brand new ID)
    scores_new, ids_new = scann(tf.constant(["brand_new_user_999"]))
    
    # 3. Format results (take top 'limit')
    old_recs = []
    for pid in ids_old.numpy().flatten()[:limit]:
        if isinstance(pid, bytes):
            old_recs.append(pid.decode('utf-8'))
        else:
            old_recs.append(str(pid))
    
    new_recs = []
    for pid in ids_new.numpy().flatten()[:limit]:
        if isinstance(pid, bytes):
            new_recs.append(pid.decode('utf-8'))
        else:
            new_recs.append(str(pid))
    
    # 4. Analyze Overlap
    intersection = set(old_recs).intersection(set(new_recs))
    
    print(f"Known User ({user_id}) Top 3: {old_recs[:3]}", flush=True)
    print(f"New User (Cold Start) Top 3: {new_recs[:3]}", flush=True)
    print(f"\nOverlap between Known and New User: {len(intersection)} posts", flush=True)
    
    if len(intersection) > 7:
        print("⚠️ RESULT: High Overlap.", flush=True)
    else:
        print("✅ RESULT: Personalization working.", flush=True)