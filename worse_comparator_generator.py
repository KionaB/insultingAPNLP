import logging
import re

import tabulate
import sentence_transformers
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet

model = SentenceTransformer("all-mpnet-base-v2")
logger = logging.getLogger(__name__)


def get_worse_comparator(comparator: str, scale=None):
    """Gets a comparator and optional scale to compare on, generates a stronger comparator
    @:arg str comparator: the comparator to outdo
    @:arg str scale: the scale to outdo comparator on, optional, if not provided function will return a more negative word
    @:returns str worse_comparator: a more negative comparator or the worse comparator on the scale"""
    worse_comparator = " "
    if scale is None:
        worse_comparator = get_multiple_anchor_comparator(comparator, "terrible")
    else:
        worse_comparator = get_multiple_anchor_comparator(comparator, scale)
    return worse_comparator


# TODO add PCA stuff

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
    syns = wordnet.synsets(word)
    synonyms = []
    hypers = []
    for syn in syns:
        for l in syn.lemmas():
            synonyms.append(l.name())
        hypers.append(syn.hypernyms())
    moresyns = []
    print("finding hypernyms")
    for h in hypers:
        for l in h:
            print(l.name)
            moresyns.append(l.name())
            for o in l.hyponyms():
                moresyns.append(o.name())
                # TODO filter out toxic words
    synonyms += moresyns
    return synonyms


def get_multiple_anchor_comparator(comparator: str, scale: str):
    """Gets comparator and scale, then uses Multiple_word_anchor.ipynb's method to generate a worse comparator based on synonyms and antonyms from nltk"""
    syns, ants = get_scale_syns_and_opposites(scale)
    words_for_comparator = get_close(comparator)
    if not ants:
        # emergency antonyms
        ants.append("amazing")
        ants.append("cool")
        # ants.append("cute")
        ants.append("smart")
    comparator_list = make_scale_list(list(set(syns)), list(set(ants)), list(set(words_for_comparator)))
    result = comparator_list[0]
    print(result)
    if ".0" in result:
        # filter out the wordnet variables to get just the word
        result = result.replace(".n.", "")
        result = result.replace(".v.", "")
        result = result.replace(".a.", "")
        result = result.replace(".s.", "")
        result = result.replace(".r.", "")
        result = re.sub(re.compile(r"[0-9]"), "", result)
        print(result)
    return re.sub("_", " ", result)  # add in spaces


# MULTIPLE_WORD_ANCHOR methods
# This combines multiple similar words into a single word anchor to remove noise
def encode_anchor(words):
    logger.info('Encoding anchor words' + str(words))
    vecs = model.encode(words)
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
        deter = model.encode(word)
        deter = deter / np.linalg.norm(deter)

        d, proj, t = proj_meas(vec1, vec2, deter)
        scale_scores.append(t)
        dist_scores.append(d)

    scores, words, dists = zip(*sorted(zip(scale_scores, word_list, dist_scores)))
    # normed_scores = (scores-min(scores))/(max(scores)-min(scores))
    normed_dists = 1 - (dists - min(dists)) / (max(dists) - min(dists))  # Normalized makes a bit more sense here
    # Build table
    table = []
    for word, score, dist, nor_dist in zip(words, scores, dists, normed_dists):
        table.append([word, f"{score:.3f}", f"{dist:.3f}", f"{nor_dist:.3f}"])

    headers = ["Word", "t (scale)", "Distance", "Normalized Distance"]
    print('From ', words1, ' to ', words2, ':')
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    return words
