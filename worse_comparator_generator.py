import logging
import re
import tabulate
import sentence_transformers
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from transformers.utils.logging import disable_progress_bar
from tabulate import tabulate
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet

model = SentenceTransformer("all-mpnet-base-v2")
logger = logging.getLogger(__name__)

#TODO expand
toxic = ["retard", "homosexual","nazi","bitch","whore","slut",
         "swastika","Hakenkreuz","Star of David","Shield of David",
         "Solomon's seal","Paschal Lamb","Mogen David","Magen David",
         "Agnus Dei","hammer and sickle","color-blind person",
         "White person","simpleton","segregate","cross-dresser",
         "weight gainer","wuss","Caucasian","religious person",
         "pussycat","drug user","Israelite","Native American","jerk",
         "nonreligious person","visually impaired person","homo",
         "person of color","transsexual","gay","mollycoddler",
         "homophile","mixed-blood","Black person","masturbator",
         "sex symbol","person of colour","mestizo","Black","Jew",
         "transexual","heterosexual","pansexual","heterosexual person",
         "ethnic","Slav","Amerindian","handicapped person","Hebrew",
         "substance abuser","transvestite","deaf person","Negroid",
         "Negro","baby buster","primitive person","aborigine","sex object",
         "aboriginal","African","misogamist","blackamoor","anti-American",
         "dyslectic"]

def get_worse_comparator(comparator: str, scale=None):
    """Gets a comparator and optional scale to compare on, generates a stronger comparator
    @:arg str comparator: the comparator to outdo
    @:arg str scale: the scale to outdo comparator on, optional, if not provided function will return a more negative word
    @:returns str worse_comparator: a more negative comparator or the worse comparator on the scale"""
    worse_comparator = " "
    disable_progress_bar()
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
    #get all the synsets that are nouns
    syns = wordnet.synsets(word,pos='n')
    synonyms = []
    hypers = []
    for syn in syns:
        for l in syn.lemmas():
            synonyms.append(l.name())
        hypers.append(syn.hypernyms())
    moresyns = []
    for h in hypers:
        for l in h:
            for lems in l.lemmas():
                moresyns.append(lems.name())
            for o in l.hyponyms():
                for p in o.lemmas():
                    moresyns.append(p.name())
    synonyms += moresyns
    # add in the spaces
    synonyms = [re.sub("_", " ", x) for x in synonyms]
    # filter out toxic and original word
    synonyms = [x for x in synonyms if (x!=word) and (x not in toxic)]
    return synonyms


def get_multiple_anchor_comparator(comparator: str, scale: str):
    """Gets comparator and scale, then uses Multiple_word_anchor.ipynb's method to generate a worse comparator based on synonyms and antonyms from nltk"""
    syns, ants = get_scale_syns_and_opposites(scale)
    words_for_comparator = get_close(comparator)
    if not ants:
        logger.warn('No antonyms found for scale ' + str(scale))
        # emergency antonyms
        ants.append("amazing")
        ants.append("cool")
        # ants.append("cute")
        ants.append("smart")
    comparator_list = make_scale_list(list(set(syns)), list(set(ants)), list(set(words_for_comparator)))
    result = comparator_list[0]
    logger.info('Worse comparator ' + str(result)+' for scale '+str(scale)+' with original comparator '+str(comparator))
    return result


# MULTIPLE_WORD_ANCHOR methods
# This combines multiple similar words into a single word anchor to remove noise
def encode_anchor(words):
    logger.info('Encoding anchor words' + str(words))
    vecs = model.encode(words,show_progress_bar=False)
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
        deter = model.encode(word,show_progress_bar=False)
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
