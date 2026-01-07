from nltk.corpus import wordnet
import fasttext
import fasttext.util
import numpy as np
import faiss

def get_scale_syns_and_opposites(scale: str, mode='wordnet'):
    """Gets a scale and returns a list of synonyms and a list of antonyms
    @:arg str scale: the scale to get synonyms and antonyms of
    @returns list of synonyms and list of antonyms"""
    def find_syn_ant_soft(scale, num_neighbours=10):
        model = fasttext.load_model('cc.en.300.bin')
        neighbours = [w for _, w in model.get_nearest_neighbors(scale, num_neighbours)]
        antonyms = []
        synonyms = []
        for nb in neighbours:
            for syn in wordnet.synsets(nb):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
                    for ant in lemma.antonyms():
                        antonyms.append(ant.name())
        if not antonyms:
            return synonyms, ['amazing', 'cool', 'smart'], False  # no antonym found
        else:
            return synonyms, list(set(antonyms)), True

    if mode == 'wordnet':
        syns = wordnet.synsets(scale)
        antonyms = []
        synonyms = []
        ants_found = True
        for syn in syns:
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        antonyms = list(set(antonyms)) # Remove duplicates
        if not antonyms:
            ants_found = False
            # emergency antonyms
            antonyms.append("amazing")
            antonyms.append("cool")
            antonyms.append("smart")
        return synonyms, antonyms, ants_found

    elif mode == 'fasttext':
        return find_syn_ant_soft(scale)

    elif mode == 'extremes':
        def extreme_neighbors(words1, words2, model, scale=10.0, k=10):
            # word vectors
            v1 = np.mean([model.get_word_vector(w) for w in words1], axis=0)
            v2 = np.mean([model.get_word_vector(w) for w in words2], axis=0)

            # direction
            direction = v2 - v1

            # two extreme points
            extreme_neg = v1 - scale * direction
            extreme_pos = v2 + scale * direction

            # --- Build FAISS cosine index ---
            words = model.get_words()
            vectors = np.vstack([model.get_word_vector(w) for w in words])

            # normalize for cosine similarity
            vecs_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            index = faiss.IndexFlatIP(vecs_norm.shape[1])
            index.add(vecs_norm)

            # normalize queries
            q_neg = extreme_neg / np.linalg.norm(extreme_neg)
            q_pos = extreme_pos / np.linalg.norm(extreme_pos)

            # search
            Dn, In = index.search(q_neg.reshape(1, -1), k)
            Dp, Ip = index.search(q_pos.reshape(1, -1), k)

            neg_neighbors = [(words[i], float(Dn[0][j])) for j, i in enumerate(In[0])]
            pos_neighbors = [(words[i], float(Dp[0][j])) for j, i in enumerate(Ip[0])]

            return neg_neighbors, pos_neighbors

        model = fasttext.load_model('cc.en.300.bin')
        syns, ants, _ = find_syn_ant_soft(scale)
        found = True
        if not ants:
            ants = ['amazing', 'cool', 'smart']
            found = False
        neg_end, pos_end = extreme_neighbors(syns, ants, model, scale=10, k=30)
        # clean_neg = dedupe_word_list_strong(neg_end, model)
        # clean_pos = dedupe_word_list_strong(pos_end, model)
        return neg_end, pos_end, found
    else:
        raise Exception("your mode was: ", mode, ".Mode must be 'wordnet', 'fasttext', or 'extremes'")