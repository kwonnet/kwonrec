
import random
from .config import  MODEL_SERVING_PATH, MAPPINGS_PATH, SCANN_INDEX_PATH, USER_EMB_PREFIX, REDIS_HOST, REDIS_PORT
import tensorflow as tf
import tensorflow_recommenders as tfrs
from datetime import datetime
import numpy as np
import redis
from .utils import refresh_user_embedding
import json
from transformers import pipeline, Pipeline
from typing import List
from keybert import KeyBERT
from collections import Counter
import re
import torch
import os
import spacy
from spacy.cli import download
from spacy.matcher import PhraseMatcher
from sentence_transformers import CrossEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)  # use 

FEED_CACHE_KEY = "feed:list:{user_id}"      # Redis list of recommendations
SEEN_CACHE_KEY = "seen:set:{user_id}"       # Redis set of seen post IDs
CACHE_TTL = 900  # 15 minutes

def generate_and_cache_feed(user_id: str, total_size: int = 300):
    print(f"Generating fresh feed for {user_id} (size={total_size})")

    # Load ScaNN once globally
    global scann_index
    if 'scann_index' not in globals():
        scann_index = tf.saved_model.load(SCANN_INDEX_PATH)
        print("ScaNN index loaded")

    # Get large candidate pool from ScaNN
    user_input = tf.constant([user_id])
    scores_raw, post_ids_raw = scann_index(user_input)

    # Decode and take top N
    post_ids = [
        pid.decode('utf-8') if isinstance(pid, bytes) else str(pid)
        for pid in post_ids_raw.numpy().flatten()[:total_size * 2]  # oversample
    ]
    scores = scores_raw.numpy().flatten().tolist()[:total_size * 2]

    # Optional: fetch latest posts for freshness
    latest_posts = [] #fetch_latest_posts(limit=30)
    latest_ids = []
    if len(latest_posts) > 0:
        latest_ids = [p['post_id'] for p in latest_posts.to_dict('records') if p['post_id'] not in post_ids[:total_size]]

    # Combine: ScaNN top + some latest
    final_ids = post_ids[:total_size - len(latest_ids)] + latest_ids
    final_ids = final_ids[:total_size]  # cap

    recommendations = [{"id": pid} for pid in final_ids]

    # Clear old cache
    feed_key = FEED_CACHE_KEY.format(user_id=user_id)
    seen_key = SEEN_CACHE_KEY.format(user_id=user_id)
    redis_client.delete(feed_key)
    redis_client.delete(seen_key)

    # Cache new feed
    if recommendations:
        redis_client.rpush(feed_key, *[json.dumps(rec) for rec in recommendations])
        redis_client.expire(feed_key, CACHE_TTL)

    print(f"Cached {len(recommendations)} posts for {user_id}")

def get_recommendations(user_id: str, limit: int = 50, refresh: bool = False):
    feed_key = FEED_CACHE_KEY.format(user_id=user_id)
    seen_key = SEEN_CACHE_KEY.format(user_id=user_id)

    print(f"Recommend user - {user_id} with limit {limit}")

    # Force refresh or cache empty
    if refresh or redis_client.llen(feed_key) == 0:
        generate_and_cache_feed(user_id, total_size=300)

    # Get seen posts
    seen = redis_client.smembers(seen_key)
    seen = {item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in seen}

    # Consume from cache
    served = []
    while len(served) < limit:
        raw = redis_client.lpop(feed_key)
        if not raw:
            # Cache exhausted → regenerate
            generate_and_cache_feed(user_id, total_size=300)
            continue
        
        rec = json.loads(raw)
        if rec["id"] not in seen:
            served.append(rec)
            redis_client.sadd(seen_key, rec["id"])
            redis_client.expire(seen_key, CACHE_TTL)
        
        if redis_client.llen(feed_key) == 0:
            break  # no more

    print(f"Served {len(served)} new posts to {user_id}")

    return served

# MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# def get_classifier():
#     if not hasattr(get_classifier, "_model"):
#         get_classifier._model = pipeline(
#             "zero-shot-classification",
#             model=MODEL_NAME,
#             device=-1
#         )
#     return get_classifier._model

# MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# MODEL_NAME = "facebook/bart-large-mnli"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


_classifier: Pipeline | None = None

def get_classifier() -> Pipeline | None:
    """
    Loads the classifier once.
    Returns None if unavailable (graceful degradation).
    """
    global _classifier

    # Already loaded
    if _classifier is not None:
        return _classifier

    try:
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            # device=0,                # set to 0 for GPU
            # local_files_only=True     # offline-safe
        )
        return _classifier

    except Exception as e:
        return None


def load_stopwords(file_path: str) -> set:
    with open(file_path, "r", encoding="utf-8") as f:
        # Strip whitespace and convert to lowercase
        stopwords = {line.strip().lower() for line in f if line.strip()}
    return stopwords


kw_model = KeyBERT("all-MiniLM-L6-v2")
# kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
# kw_model = KeyBERT("distilbert-base-nli-mean-tokens")


STOPWORDS_FILE = os.path.join(BASE_DIR, "stopwords.txt")


STOPWORDS = load_stopwords(STOPWORDS_FILE)

def has_non_stopword(phrase: str) -> bool:
    words = phrase.split()
    return any(w not in STOPWORDS for w in words)

PREPOSITIONS = {"of", "in", "on", "for", "with", "as", "by", "at", "from"}

CONJUNCTIONS = {"and", "but", "or", "while", "so"}



# download model if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def has_content_word(phrase: str) -> bool:
    words = [w.lower() for w in phrase.split() if w.lower() not in STOPWORDS]
    return len(words) >= 2

def has_noun(phrase: str) -> bool:
    doc = nlp(phrase)
    return any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc)

def is_valid_phrase(phrase: str) -> bool:
    words = phrase.lower().split()
    
    # dangling start/end
    if words[0] in CONJUNCTIONS or words[-1] in PREPOSITIONS:
        return False
    
    # minimum content words
    if not has_content_word(phrase):
        return False
    
    # must contain a noun/proper noun
    if not has_noun(phrase):
        return False
    
    return True

def filter_wrong_phrases(keywords):
    return [kw for kw in keywords if is_valid_phrase(kw["phrase"])]


from collections import defaultdict, Counter
from itertools import combinations

def build_cooccurrence(keywords, window=2):
    """
    Build co-occurrence counts from extracted phrases.
    """
    cooc = defaultdict(Counter)
    for k in keywords:
        words = k["phrase"].lower().split()
        for i in range(len(words)):
            for j in range(i+1, min(i+window+1, len(words))):
                cooc[words[i]][words[j]] += 1
                cooc[words[j]][words[i]] += 1
    return cooc


def phrase_coherence_score(phrase, cooc):
    words = phrase.lower().split()
    if len(words) < 2:
        return 0
    score = 0
    pairs = list(combinations(words, 2))
    for w1, w2 in pairs:
        score += cooc[w1].get(w2, 0)
    return score / len(pairs)  # average co-occurrence



def restore_phrases_casing(original_text: str, extracted_keywords: list[dict]) -> list[dict]:
    """
    Restores capitalization for words in extracted keyword phrases based on original text.
    Keeps the score intact.
    """
    # Collect all words that have uppercase letters in the original text
    words_with_caps = set(re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', original_text))

    restored_keywords = []
    for kw in extracted_keywords:
        phrase = kw["phrase"]
        words = phrase.split()
        # Replace words if they appear in the original capitalized words
        restored_words = [
            next((orig for orig in words_with_caps if orig.lower() == w.lower()), w)
            for w in words
        ]
        restored_keywords.append({
            "phrase": " ".join(restored_words),
            "score": kw["score"]
        })
    
    return restored_keywords


def extract_entities(text):
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return list(set(persons))


content = """Music has no borders, and artists like Davido, Eminem, and Nicki Minaj prove that impact isn’t limited by geography or genre. Davido represents the global rise of Afrobeats, blending African rhythm with mainstream appeal. Eminem remains one of the most technically gifted lyricists in hip-hop history, known for raw storytelling and unmatched wordplay. Nicki Minaj stands as a cultural icon, reshaping female rap with versatility, confidence, and chart-dominating records.Despite coming from different worlds, all three artists share one thing in common: influence. Their music travels across continents, shapes pop culture, and inspires millions of fans worldwide. 2baba has been driving the african music too. Listen to 2Baba's new album, it's fire! 50Cent and Nicki Minaj collaborated again. 
001 Records signed Peter Kelvin Torver. 21Savage dropped a surprise track. 
The Beatles' are legends but 3DoorsDown had some hits too. A$AP Rocky is now married to Rihanna. A$AP Rocky just dropped a new album! $bill is making waves. #Flash is trending worldwide. 2baba, Annie were also present at the event. The Flash actor was Grant Gustin and he was epic. A Tribe Called Quest was at the concert turining it up too
"""


nlp = spacy.load("en_core_web_sm")

def remove_hashtags(text: str) -> str:
    return re.sub(r"\s*#\w+", "", text).strip()

def extract_relevant_keywords(text):
    # 1. Capture Hashtags
    hashtags = re.findall(r'#\w+', text)
    
    # 2. Prepare text for NLP (removing tags to prevent clumping)
    text_no_tags = re.sub(r'#\w+', '', text)
    doc = nlp(text_no_tags)
    
    candidates = set()

    # 3. Extract Proper Nouns (Surgical Trim)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "PRODUCT", "GPE"]:
            # Strip verbs/leading junk
            filtered = [t.text for t in ent if t.pos_ in ["PROPN", "NOUN"]]
            if filtered:
                candidates.add(" ".join(filtered).strip())

    # 4. Extract Long Clauses & Slang
    # We split by punctuation to catch phrases like "Dey for who dey for you"
    clauses = re.split(r'[,.!?;]\s*', text_no_tags)
    for clause in clauses:
        clean_clause = clause.strip()
        word_count = len(clean_clause.split())
        
        # We target phrases that are 3+ words or contain specific slang markers
        if word_count >= 3:
            # Avoid long generic sentences by ensuring it's not a full paragraph
            if word_count < 12: 
                candidates.add(clean_clause)

    # 5. Deduplication (Keep Shorter for Entities, Keep Longer for Phrases)
    sorted_candidates = sorted(list(candidates), key=len)
    unique_phrases = []

    for item in sorted_candidates:
        # If it's a short entity (1-2 words), keep the shortest version
        if len(item.split()) <= 2:
            if not any(existing.lower() in item.lower() for existing in unique_phrases):
                unique_phrases.append(item)
        else:
            # For long idiomatic phrases, we keep them if they aren't exact duplicates
            if not any(item.lower() == existing.lower() for existing in unique_phrases):
                unique_phrases.append(item)

    # 6. Format Output
    final_output = []
    # Merge hashtags and phrases
    all_signals = list(set(hashtags) | set(unique_phrases))
    
    for val in sorted(all_signals):
        if val:
            final_output.append({
                "phrase": val,
                "score": round(random.uniform(0.6, 1.0), 10)
            })

    return final_output



def extract_proper_words(text):
    # Match hashtags or capitalized/digit-starting multi-word phrases
    pattern = r'''
    (
        \b
        (?:[\d$][A-Za-z0-9$@'-]*|[A-Z][A-Za-z0-9$@'-]*)
        (?:\s+(?:[\d$][A-Za-z0-9$@'-]*|[A-Z][A-Za-z0-9$@'-]*)){0,5}
        \b
    )
    |
    (
        \#[A-Za-z0-9_]+
    )
    '''

    matches = re.findall(pattern, text, re.VERBOSE)

    canonical = {}

    for group_a, group_b in matches:
        name = (group_a or group_b).strip()
        if len(name) <= 2:
            continue

        # 1. Remove trailing apostrophe / possessive
        name = re.sub(r"(?:['’]s?|['’])$", "", name)

        # 2. Strip leading stopword (multi-word only)
        words = name.split()
        while len(words) > 1 and words[0].lower() in STOPWORDS and not name.startswith('#'):
            words = words[1:]
            name = ' '.join(words)

        # 3. Drop single-word stopwords (non-hashtag)
        if len(words) == 1 and not name.startswith('#') and words[0].lower() in STOPWORDS:
            continue

        # 4. Casing unification
        key = name.lower()
        if key not in canonical:
            canonical[key] = name
        else:
            existing = canonical[key]
            if sum(c.isupper() for c in name) > sum(c.isupper() for c in existing):
                canonical[key] = name

    # Final output
    proper_names = [
        {"phrase": v, "score": round(random.uniform(0.6, 1.0), 10)}
        for v in sorted(canonical.values(), key=len, reverse=True)
    ]

    return proper_names

def merge_and_refine(*lists):
    # 1. Flatten all lists into one
    combined = [item for sublist in lists for item in sublist]
    
    # 2. Sort by score descending
    # This ensures that when we remove duplicates, we keep the one with the highest score
    combined.sort(key=lambda x: x['score'], reverse=True)
    
    unique_results = {}
    
    for item in combined:
        # Normalize the phrase to lowercase for comparison
        # This treats "Eminem", "eminem", and "EMINEM" as the same thing
        phrase_key = item['phrase'].strip().lower()
        
        # Only add if we haven't seen this phrase (ignoring case) yet
        if phrase_key not in unique_results:
            unique_results[phrase_key] = item
            
    # 3. Return the results as a list, sorted alphabetically
    return sorted(unique_results.values(), key=lambda x: x['phrase'].lower())



def simple_stopword_filter(keyword_array):
    filtered_results = []
    
    for item in keyword_array:
        phrase = item['phrase'].strip()
        words = phrase.lower().split()
        
        # Rule: Any phrase with length > 2 words 
        # that ends with a STOPWORD should be filtered out
        if len(words) > 2:
            if words[-1] in STOPWORDS:
                continue  # Skip this phrase
        
        # Rule: Let's also catch the 2-word fragments that end in stopwords
        # e.g., "concert was", "Minaj and"
        if len(words) == 2:
            if words[-1] in STOPWORDS:
                continue
                
        filtered_results.append(item)
        
    return filtered_results


def refine_final_keywords(keyword_array):
    refined = []
    
    # 1. POS Trimming (Removes "Wish", "It's", "And", etc.)
    for item in keyword_array:
        phrase = item['phrase']
        if phrase.startswith("#"):
            refined.append(item)
            continue
            
        doc = nlp(phrase)
        start_idx = 0
        for token in doc:
            # Trim leading Verbs, Pronouns, and Conjunctions
            if token.pos_ in ["VERB", "PRON", "DET", "AUX", "PART", "CCONJ"]:
                start_idx += 1
            else:
                break
        
        clean_phrase = " ".join([t.text for t in doc[start_idx:]]).strip()
        if clean_phrase:
            refined.append({"phrase": clean_phrase, "score": item['score']})

    # 2. Score-based Case-Insensitive Deduplication
    unique_map = {}
    for item in refined:
        key = item['phrase'].lower()
        if key not in unique_map or item['score'] > unique_map[key]['score']:
            unique_map[key] = item

    # 3. Substring Logic: Keep longer phrases IF they add a NOUN
    sorted_items = sorted(unique_map.values(), key=lambda x: len(x['phrase']))
    final = []
    
    for item in sorted_items:
        is_redundant = False
        item_phrase_low = item['phrase'].lower()
        
        for existing in final:
            existing_phrase_low = existing['phrase'].lower()
            
            # If "christmas" is already in, and current is "christmas season"
            if existing_phrase_low in item_phrase_low:
                # Find the words that are new
                new_words = [w for w in item['phrase'].split() 
                             if w.lower() not in existing_phrase_low]
                
                # Check if any of the new words are Nouns/Proper Nouns
                new_doc = nlp(" ".join(new_words))
                has_new_noun = any(t.pos_ in ["NOUN", "PROPN"] for t in new_doc)
                
                # If it's just fluff (e.g., "Davido" -> "Davido was"), it's redundant.
                # If it adds a noun (e.g., "Davido" -> "Davido concert"), it is NOT redundant.
                if not has_new_noun:
                    is_redundant = True
                    break
        
        if not is_redundant:
            final.append(item)

    return sorted(final, key=lambda x: x['score'], reverse=True)

# 2. Initialize Cross-Encoder (The Judge)
# This model is specifically trained to rank relevance
ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rank_keywords(text, keywords):
    pairs = [[text, kw[0]] for kw in keywords]
    
    # Passing activation_fct converts raw logits to 0-1 probabilities
    relevance_scores = ranker.predict(pairs, activation_fct=torch.nn.Sigmoid())
    
    refined = []
    for i, score in enumerate(relevance_scores):
        refined.append({
            "phrase": keywords[i][0],
            "score": float(score)  # Now this will be between 0 and 1
        })
        
    return sorted(refined, key=lambda x: x['score'], reverse=True)



def get_content_keywords(content: str):
    try:
        # Get proper words
        proper_words = extract_proper_words(content)
        # Get relevant words
        relevant_words = extract_relevant_keywords(content)
        # clean and remove all hashtags
        clean_text = remove_hashtags(content)
        # extract keywords from model
        keywords = kw_model.extract_keywords(
            clean_text,
            keyphrase_ngram_range=(1, 5),
            stop_words="english",
            top_n=100,
            use_mmr=True, 
            diversity=0.4,
            # nr_candidates=50,
        )
        # rank keywords
        keywords = rank_keywords(clean_text, keywords)

        # Output top 5 "Gold" keywords
        print(keywords[:5])

        # keywords = [{"phrase": p, "score": s} for p, s in keywords if has_non_stopword(p)]

        keywords = restore_phrases_casing(content, keywords)

        # keywords = merge_and_refine(relevant_words, proper_words, keywords)

        # keywords = simple_stopword_filter(keywords)

        # keywords = refine_final_keywords(keywords)

        # merge keywords with proper words
        # keywords = keywords.extend(proper_words)


        # keywords = filter_wrong_phrases(keywords)

        # logger.info(f"total len after filter {len(keywords)}")

        # cooc = build_cooccurrence(keywords)
        # threshold = 0.6  # tune this based on your data

        # keywords = [
        #     k for k in keywords
        #     if phrase_coherence_score(k["phrase"], cooc) >= threshold
        # ]

        # logger.info(f"total len after coherence {len(keywords)}")


        return keywords

    except Exception as e:
        raise e