import logging
import re
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from nltk.corpus import wordnet
import random
import pickle
import torch

if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"

model = SentenceTransformer("all-mpnet-base-v2")
logger = logging.getLogger(__name__)


def get_worse_comparator(comparator: str, method: bool, scale=None):
    """Gets a comparator and optional scale to compare on, generates a stronger comparator
    @:arg str comparator: the comparator to outdo
    @:arg str scale: the scale to outdo comparator on, optional, if not provided function will return a more negative word
    @:returns str worse_comparator: a more negative comparator or the worse comparator on the scale"""
    worse_comparator = " "
    if scale is None:
        worse_comparator = get_multiple_anchor_comparator(comparator, "terrible", method)
    else:
        worse_comparator = get_multiple_anchor_comparator(comparator, scale, method)
    return worse_comparator

def get_scale_syns_and_opposites(scale: str):
    """Gets a scale and returns a list of synonyms and a list of antonyms
    @:arg str scale: the scale to get synonyms and antonyms of
    @returns list of synonyms and list of antonyms"""
    syns = wordnet.synsets(scale)
    antonyms = []
    synonyms = []
    for syn in syns:
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return synonyms, antonyms

# TODO make list option for words

def get_close(word):
    """Gets a list of nearby words
    @arg str word: the word to get synonyms for
    @returns list of nearby words"""
    # TODO limit options to nouns?
    word = word.replace(" ", "_")
    syns = wordnet.synsets(word)
    synonyms = []
    hypers = []
    for syn in syns:
        for l in syn.lemmas():
            synonyms.append(l.name())
        hypers.append(syn.hypernyms())
    moresyns = []
    # print("finding hypernyms")
    for h in hypers:
        for l in h:
            # print(l.name)
            moresyns.append(l.name())
            for o in l.hyponyms():
                moresyns.append(o.name())
                # TODO filter out toxic words
    synonyms += moresyns
    return synonyms

def get_multiple_anchor_comparator(comparator: str, scale: str, method: bool):
    """Gets comparator and scale, then uses Multiple_word_anchor.ipynb's method to generate a worse comparator based on synonyms and antonyms from nltk"""
    syns, ants = get_scale_syns_and_opposites(scale)
    words_for_comparator = get_close(comparator)
    if not ants:
        # emergency antonyms
        ants.append("amazing")
        ants.append("cool")
        ants.append("smart")
    if method: 
        comparator_list = pca(list(set(syns)), list(set(ants)), list(set(words_for_comparator)))
    else:
        comparator_list = make_scale_list(list(set(syns)), list(set(ants)), list(set(words_for_comparator)))
    index = random.randint(0, len(comparator_list)-1)
    result = comparator_list[index]
    # print(result)
    if ".0" in result:
        # filter out the wordnet variables to get just the word
        for var in [".n.", ".v.", ".a.", ".s.", ".r."]:
            result = result.replace(var, "")
        result = re.sub(re.compile(r"[0-9]"), "", result)
        # print(result)
    return re.sub("_", " ", result)  # add in spaces


# MULTIPLE_WORD_ANCHOR methods ---------------------------------------------------------
# This combines multiple similar words into a single word anchor to remove noise
def encode_anchor(words):
    logger.info('Encoding anchor words' + str(words))
    vecs = model.encode(words, show_progress_bar=False)
    vecs = normalize(vecs)
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
def make_scale_list(words1, words2, word_list):
    scale_scores = []
    dist_scores = []
    vec1 = encode_anchor(words1)
    vec2 = encode_anchor(words2)

    for word in word_list:
        deter = model.encode(word, show_progress_bar=False)
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
    return words

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
    pca = PCA(n_components=1)
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
    print(tabulate(table,
                   headers=["Word", "t (PC1 position)", "Orthogonal distance"],
                   tablefmt="fancy_grid"))

    # Return only the sorted word list
    sorted_words = [w for (w, _, _) in results]
    return sorted_words
