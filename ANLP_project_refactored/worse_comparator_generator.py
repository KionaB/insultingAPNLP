import logging
import re
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils.logging import disable_progress_bar
from tabulate import tabulate
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from nltk.corpus import wordnet
import random
import pickle
import torch

# We can make these more specific later
from syn_ant_generation import *
from word_list_gen import *
# End of prev comment

if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"

model = SentenceTransformer("all-mpnet-base-v2")
logger = logging.getLogger(__name__)

def get_worse_comparator(syns, ants, words_for_comparator, method: bool, pca_method=False, mid_adjust=False, vec_model='wordnet'):
    """Gets a comparator and optional scale to compare on, generates a stronger comparator
    @:arg str comparator: the comparator to outdo
    @:arg str scale: the scale to outdo comparator on, optional, if not provided function will return a more negative word
    @:returns str worse_comparator: a more negative comparator or the worse comparator on the scale"""
    worse_comparator = " "
    disable_progress_bar()
    # print(syns, ants, words_for_comparator, method, pca)
    if not pca_method:
        worse_comparator, scores, dists = make_scale_list(syns, ants, words_for_comparator, vec_model)
    else:
        worse_comparator, scores, dists = pca(syns, ants, words_for_comparator)

    if mid_adjust:
        normed_dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
        t_avg = np.mean(scores)
        scores = scores + (t_avg - scores) * normed_dists
    return worse_comparator, scores

# TODO make list option for words

# def get_multiple_anchor_comparator(comparator: str, scale: str, method: bool):
# def get_multiple_anchor_comparator(syns, ants, words_for_comparator, method: bool):
#     """Gets comparator and scale, then uses Multiple_word_anchor.ipynb's method to generate a worse comparator based on synonyms and antonyms from nltk"""
#     if method:
#         comparator_list, scores = pca(list(set(syns)), list(set(ants)), list(set(words_for_comparator)))
#     else:
#         comparator_list, scores = make_scale_list(list(set(syns)), list(set(ants)), list(set(words_for_comparator)), vec_model)
#     #TODO: make the index higher than current used word and make sure to not use already used words
#     # index = random.randint(0, len(comparator_list)-1)
#     return comparator_list, scores

    # index = 0
    # result = comparator_list[index]
    # # print(result)
    # if ".0" in result:
    #     # filter out the wordnet variables to get just the word
    #     for var in [".n.", ".v.", ".a.", ".s.", ".r."]:
    #         result = result.replace(var, "")
    #     result = re.sub(re.compile(r"[0-9]"), "", result)
    #     # print(result)
    # return re.sub("_", " ", result)  # add in spaces


# MULTIPLE_WORD_ANCHOR methods ---------------------------------------------------------
# This combines multiple similar words into a single word anchor to remove noise
def encode_anchor(words, vec_model='wordnet'):
    if vec_model == 'wordnet':
        logger.info('Encoding anchor words' + str(words))
        vecs = model.encode(words, show_progress_bar=False)
        vecs = normalize(vecs) # I still find this step a bit suspiscious
        mean_vec = np.mean(vecs, axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        return mean_vec
    elif vec_model == 'fasttext':
        logger.info('Encoding anchor words' + str(words))
        new_model = fasttext.load_model('cc.en.300.bin')
        vecs = np.array([new_model.get_word_vector(w) for w in words])
        vecs = normalize(vecs)  # I still find this step a bit suspiscious
        mean_vec = np.mean(vecs, axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        return mean_vec

# This function does the projection
def proj_meas(v1, v2, v3):
    v = v2 - v1
    w = v3 - v1
    proj = np.dot(w, v) / np.dot(v, v) * v
    d = np.linalg.norm(w - proj)
    t = np.dot(w, v) / np.dot(v, v)  # t is how far along on the spectrum something is, 0.0 for v1, 1.0 for v2.
    proj_point = v1 + t * v
    return d, proj_point, t


# Here I make the list and sort them based on the scale
def make_scale_list(words1, words2, word_list, vec_model='wordnet'):
    scale_scores = []
    dist_scores = []
    vec1 = encode_anchor(words1, vec_model=vec_model)
    vec2 = encode_anchor(words2, vec_model=vec_model)
    if vec_model == 'fasttext':
        new_model = fasttext.load_model('cc.en.300.bin')
    for word in word_list:
        if vec_model == 'wordnet':
            deter = model.encode(word, show_progress_bar=False)
        elif vec_model == 'fasttext':
            deter = new_model.get_word_vector(word)
        else:
            raise Exception("You did not input a correct model, please pick 'wordnet' or 'fasttext', you entered: ", vec_model)
        deter = deter / np.linalg.norm(deter)

        d, proj, t = proj_meas(vec1, vec2, deter)
        scale_scores.append(t)
        dist_scores.append(d)

    scores, words, dists = zip(*sorted(zip(scale_scores, word_list, dist_scores)))
    normed_dists = 1 - (dists - min(dists)) / (max(dists) - min(dists))  # Normalized makes a bit more sense here
    # Build table
    table = []
    for word, score, dist, nor_dist in zip(words, scores, dists, normed_dists):
        table.append([word, f"{score:.3f}", f"{dist:.3f}", f"{nor_dist:.3f}"])

    headers = ["Word", "t (scale)", "Distance", "Normalized Distance"]
    # print('From ', words1, ' to ', words2, ':')
    # print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    return words, scores, dists






# PRINCIPLE COMPONENT ANALYSIS METHOD ------------------------------------------------------
def build_pca_axis(pos_words, neg_words, model):
    # Maak een unieke 'fingerafdruk' van je input om te zien of de semantic scale verandert
    scale_hash = hash(tuple(pos_words) + tuple(neg_words))

    # Encode synonyms/antonyms woorden
    pos_vecs = model.encode(pos_words, convert_to_numpy=True, device=device)
    neg_vecs = model.encode(neg_words, convert_to_numpy=True, device=device)

    # Stack de twee matrixen op elkaar
    X = np.vstack([pos_vecs, neg_vecs])

    # doe de daadwerkelijke PCA berekening waarbij covariance matrx en richting van alle anchors worden berekend.
    pca = PCA(n_components=1, svd_solver='auto', whiten=False)
    pca.fit(X)
    axis = pca.components_[0]
    axis /= np.linalg.norm(axis) # zorg opnieuw ervoor dat de PCA vector lengte 1 is.
    anchor_mean = X.mean(axis=0) # middenpunt van je anchors
    return axis, anchor_mean, scale_hash

# projecteer
def project_on_pca_axis(axis, anchor_mean, vec):
    centered = vec - anchor_mean # centreer PCA coordinated naar middenpunt van je anchor
    t = np.dot(centered, axis)
    proj_point = anchor_mean + t * axis
    d = np.linalg.norm(centered - t * axis)
    return t, d, proj_point

def pca(syns, ants, word_list):
    # Build a hash that uniquely identifies THIS semantic scale + word list
    scale_hash = hash(tuple(syns) + tuple(ants) + tuple(word_list))

    pca_cache_file = "pca_cache.pkl"
    recompute_pca = True

    # Attempt loading PCA cache
    if os.path.exists(pca_cache_file):
        with open(pca_cache_file, "rb") as f:
            cached = pickle.load(f)

        if cached["scale_hash"] == scale_hash:
            axis = cached["axis"]
            anchor_mean = cached["anchor_mean"]
            word_vecs = cached["word_vecs"]
            recompute_pca = False

    # Recompute pca if not in cache
    if recompute_pca:
        # Encode the words for this list
        word_vecs = model.encode(word_list, convert_to_numpy=True, device=device)
        word_vecs = normalize(word_vecs)

        # Compute PCA axis
        axis, anchor_mean, _ = build_pca_axis(syns, ants, model)

        # Save everything to cache
        with open(pca_cache_file, "wb") as f:
            pickle.dump({
                "scale_hash": scale_hash,
                "axis": axis,
                "anchor_mean": anchor_mean,
                "word_vecs": word_vecs
            }, f)

    # Rank words
    results = []
    for w, vec in zip(word_list, word_vecs):
        t, d, _ = project_on_pca_axis(axis, anchor_mean, vec)
        results.append((w, t, d))

    # Sort by PCA position coordinate t
    results.sort(key=lambda x: x[1])

    # Print nice table
    table = [[w, f"{t:.3f}", f"{d:.3f}"] for (w, t, d) in results]
    # print(tabulate(table,
    #                headers=["Word", "t (PC1 position)", "Orthogonal distance"],
    #                tablefmt="fancy_grid"))

    # Return only the sorted word list
    sorted_words = [w for (w, _, _) in results]
    sorted_scores = [s for (_, s, _) in results]
    sorted_dists = [d for (_, _, d) in results]
    return sorted_words, sorted_scores, sorted_dists
