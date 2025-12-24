import re
from nltk.corpus import wordnet

def get_close(word):
    """Gets a list of nearby words
    @arg str word: the word to get synonyms for
    @returns list of nearby words"""
    #get all the synsets that are nouns

    toxic = ["retard", "homosexual", "nazi", "bitch", "whore", "slut",
             "swastika", "Hakenkreuz", "Star of David", "Shield of David",
             "Solomon's seal", "Paschal Lamb", "Mogen David", "Magen David",
             "Agnus Dei", "hammer and sickle", "color-blind person",
             "White person", "simpleton", "segregate", "cross-dresser",
             "weight gainer", "wuss", "Caucasian", "religious person",
             "pussycat", "drug user", "Israelite", "Native American", "jerk",
             "nonreligious person", "visually impaired person", "homo",
             "person of color", "transsexual", "gay", "mollycoddler",
             "homophile", "mixed-blood", "Black person", "masturbator",
             "sex symbol", "person of colour", "mestizo", "Black", "Jew",
             "transexual", "heterosexual", "pansexual", "heterosexual person",
             "ethnic", "Slav", "Amerindian", "handicapped person", "Hebrew",
             "substance abuser", "transvestite", "deaf person", "Negroid",
             "Negro", "baby buster", "primitive person", "aborigine", "sex object",
             "aboriginal", "African", "misogamist", "blackamoor", "anti-American",
             "dyslectic"]

    word = word.replace(" ", "_")
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