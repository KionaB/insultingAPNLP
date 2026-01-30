from nltk.corpus import wordnet
import numpy as np
import faiss

def get_scale_syns_and_opposites(scale: str, fasttext_model, mode='wordnet'):
    """Gets a scale and returns a list of synonyms and a list of antonyms
    @:arg str scale: the scale to get synonyms and antonyms of
    @returns list of synonyms and list of antonyms"""
    
    def find_syn_ant_soft(scale, fasttext_model, num_neighbours=10):
        neighbours = [scale]+[w for _, w in fasttext_model.get_nearest_neighbors(scale, num_neighbours)]
        print(scale, neighbours, scale in neighbours)
        antonyms = []
        synonyms = []
        for nb in neighbours:
            for syn in wordnet.synsets(nb):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
                    for ant in lemma.antonyms():
                        antonyms.append(ant.name())
                    # for ant in lemma.antonyms():
                    #     for syn_ant in ant.synset().lemmas():
                    #         antonyms.append(syn_ant.name())
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
                # for ant in l.antonyms():
                #     for syn_ant in ant.synset().lemmas():
                #         antonyms.append(syn_ant.name())
        antonyms = list(set(antonyms)) # Remove duplicates
        if not antonyms:
            ants_found = False
            # emergency antonyms
            antonyms.append("amazing")
            antonyms.append("cool")
            antonyms.append("smart")
        return synonyms, antonyms, ants_found

    elif mode == 'fasttext':
        return find_syn_ant_soft(scale, fasttext_model, 10)

    elif mode == 'extremes':
        def extreme_neighbors(words1, words2, fasttext_model, scale=10.0, k=10):
            # word vectors
            v1 = np.mean([fasttext_model.get_word_vector(w) for w in words1], axis=0)
            v2 = np.mean([fasttext_model.get_word_vector(w) for w in words2], axis=0)

            # direction
            direction = v2 - v1

            # two extreme points
            extreme_neg = v1 - scale * direction
            extreme_pos = v2 + scale * direction

            # --- Build FAISS cosine index ---
            words = fasttext_model.get_words()
            vectors = np.vstack([fasttext_model.get_word_vector(w) for w in words])

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

        syns, ants, _ = find_syn_ant_soft(scale, fasttext_model, 10)
        found = True
        if not ants:
            ants = ['amazing', 'cool', 'smart']
            found = False
        neg_end, pos_end = extreme_neighbors(syns, ants, fasttext_model, scale=10, k=30)
        # clean_neg = dedupe_word_list_strong(neg_end, fasttext_model)
        # clean_pos = dedupe_word_list_strong(pos_end, fasttext_model)
        neg_end = [neg_word for neg_word, _ in neg_end]
        pos_end = [pos_word for pos_word, _ in pos_end]
        return neg_end, pos_end, found
    else:
        raise Exception("your mode was: ", mode, ".Mode must be 'wordnet', 'fasttext', or 'extremes'")