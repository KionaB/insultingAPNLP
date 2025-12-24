from nltk.corpus import wordnet

def get_scale_syns_and_opposites(scale: str):
    """Gets a scale and returns a list of synonyms and a list of antonyms
    @:arg str scale: the scale to get synonyms and antonyms of
    @returns list of synonyms and list of antonyms"""
    syns = wordnet.synsets(scale)
    antonyms = []
    synonyms = []
    ants_found = True
    for syn in syns:
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    if not antonyms:
        ants_found = False
        # emergency antonyms
        antonyms.append("amazing")
        antonyms.append("cool")
        antonyms.append("smart")
    return synonyms, antonyms, ants_found