import re
import nltk
import Levenshtein
from nltk.stem import PorterStemmer
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")


def clean_syn_ant(word_list):
    lowercase_list = [item.lower() for item in word_list]
    pattern = re.compile(r'^(?!.*([a-z])\1\1)[a-z-]+$')
    filtered = [s for s in lowercase_list if pattern.match(s)]

    stemmer = PorterStemmer()
    stems = []
    stemmed_list = []

    for word in filtered:
        stem = stemmer.stem(word.lower())

        if stem in stems:
            # Find the index of this stem
            idx = stems.index(stem)
            # Keep the shortest word
            if len(word) < len(stemmed_list[idx]):
                stemmed_list[idx] = word
        else:
            # New stem → append
            stems.append(stem)
            stemmed_list.append(word)

    levenshteinlist = []
    for word in stemmed_list:
        # Skip this word if it is too close to any already kept word
        if any(Levenshtein.distance(word, k) <=4 for k in levenshteinlist):
            continue
        levenshteinlist.append(word)

    result = list(set(levenshteinlist))
    return result

# syns_dumb = ['shitfest', 'longwindedness', 'incoherence', 'horridness', 'awfulness', 'crap-fest', 'travesty', 'maundering', 'buffoonery', 'horror-show', 'whinefest', 'suckfest', 'onanistic', 'shit-show', 'shitshow', 'pisspot', 'insipidness', 'ignominy', 'crapulous', 'farrago', 'grotesquerie', 'lunacy', 'vomitous', 'inanity', 'tastelessness', 'bastardisation', 'whine-fest', 'bastardization', 'caterwauling', 'humbuggery']
# ants_dumb = ['smart', 'intelligent', 'Smart', 'smart-', 'smarter', 'Intelligent', 'super-smart', 'ultra-smart', 'resourceful', 'smart.', 'inteligent', 'intellegent', 'smartest', 'efficient', 'sophisticated', '-Smart', 'supersmart', 'stylish', 'highly-intelligent', 'innovative', 'smar', 'Inteligent', 'not-so-smart', 'flexible', 'adaptable', 'intelligent.', '.Smart', 'energy-smart', 'forward-thinking', 'multi-functional']
#
# syns_ugly = ['horrid', 'hideous', 'ugly', 'horrible', 'vile', 'disgusting', 'horrendous', 'ghastly', 'awful', 'loathsome', 'nasty', 'terrible', 'dreadful', 'atrocious', 'wretched', 'god-awful', 'God-awful', 'despicable', 'godawful', 'execrable', 'loathesome', 'abominable', 'digusting', 'odious', 'putrid', 'revolting', 'icky', 'appalling', 'horrific', 'vomitous']
# ants_ugly = ['Bluworld', 'location.Our', 'region.Our', 'home.Features', 'Cottage', 'home.Beautiful', 'Timepieces', 'offer.Our', 'DescriptionEnjoy', 'centrally-located', 'Weddingstar', 'Located', 'Eliros', 'setting.Our', '00ZTamara', 'Picturesque', 'self-catering', 'HeartSong', 'amenities.A', 'villa', 'detailsA', 'Spacious', 'spacious', 'SongBook', '00ZRomy', 'Divonne', 'Inspirato', '00ZMathieu', '00ZModern', '00ZBruno']
#
# syns_lazy = ['lazy', 'slothful', 'work-shy', 'indolent', 'shiftless', 'dozy', 'good-for-nothing', 'lazy-ass', 'good-for-nothings', 'wastrels', 'thriftless', 'lay-about', 'dullard', 'dull-witted', 'wastrel', 'slovenly', 'oaf', 'feckless', 'lazybones', 'otiose', 'languid', 'lumpish', 'drivelling', 'workshy', 'layabouts', 'snoozy', 'indolently', 'ambitionless', 'dotard', 'gormless']
# ants_lazy = ['motivated', 'motived', 'Motivated', 'motivates', 'motivation', 'motivating', 'motivate', 'motiviated', 'motivat', 'motivated.', 'motivator', 'fueled', 'Leadercast', 'driven', 'committed', 'focused', 'motiviation', 'LeadHer', 'resonated', 'MOTIVATED', 'motivators', 'energized', 'Fueled', 'MaxPoint', 'self-motivated', 'CAMEX', 'helped', 'forefront', 'Nachawati', 'GraceWorks']
#
# syns_clueless = ['jejune', 'dimwittedness', 'simple-mindedness', 'inarticulateness', 'graceless', 'nonsensical', 'jumbled-up', 'literal-mindedness', 'bumblings', 'witlessness', 'clumsy', 'oafishly', 'catachresis', 'inelegant', 'longwindedness', 'oafishness', 'flailings', 'inelegance', 'pseudo-intellectualism', 'feckless', 'oafish', 'inconsequentiality', 'maundering', 'nincompoopery', 'jumble', 'addlepated', 'farrago', 'simplemindedness', 'blunderings', 'substance-less']
# ants_clueless = ['informed', 'competent', 'Informed', 'apprised', 'fully-informed', 'knowledgeable', 'notified', 'imformed', 'up-to-date', 'respected', 'informed.The', 'advised', 'contacted', 'empowered', 'well-informed', 'inform', 'requested', 'informing', 'qualified', 'briefed', 'infomed', 'confident', 'relevant', 'knowledgible', 'authorized', 'communicated', 'consulted', 'up-todate', 'responsive', 'proactive']
#
# syns_slow = ['dreary', 'dispiriting', 'blahness', 'drab', 'drearier', 'dour', 'dreariness', 'benighted', 'dull', 'disinteresting', 'navel-gazing', 'inconsequentiality', 'maunderings', 'drabness', 'unremembered', 'stultifying', 'maundering', 'over-written', 'dullish', 'unamusing', 'drear', 'unenlightening', 'dourness', 'snooze-fest', 'insipid', 'unrelieved', 'dullness', 'uninteresting', 'insufferable', 'dismal']
# ants_slow = ['fast', 'Fast', 'super-fast', 'FAST', 'fast.', 'fast-', 'fast.We', 'fast.It', 'fast.The', 'fastand', 'fast.This', 'fastest', 'fast.I', 'fast.What', 'fast.How', 'fast.In', 'superfast', 'quickly', 'fast.So', '-Fast', 'faster', 'Super-fast', 'fast.They', 'fast.You', 'ultra-fast', 'fast.When', 'fast.As', 'fast.And', 'non-fast', 'fastI']
#
# syns_annoying = ['Typecasting', 'Irredeemable', 'story.Anyway', 'FALs', 'non-interesting', 'Ranty', 'AMFact', 'pigshit', 'CCW', 'skidmark', 'Unavoidable', 'plot-based', 'Fuckwit', 'UXO', 'fly-posting', 'tenent', 'Featureless', 'Commie', 'per-page', 'CISC', 'last-but-one', 'Sub-standard', 'gangstalking', 'roadsigns', 'NWM', 'tabloid-esque', 'Dishonesty', 'penny-ante', 'Gatso', 'dumpsters']
# ants_annoying = ['soothe', 'sooth', 'soothes', 'soothed', 'Soothe', 'soothing', 'pacify', 'sooths', 'assuage', 'nourish', 'soother', 'rejuvenate', 'Soothes', 'invigorate', 'soothers', 'heal', 'self-soothe', 'mollify', 'calms', 'relieve', 'de-stress', 'revitalize', 'Soothed', 're-energize', 'pamper', 'calming', 'salve', 'Sooth', 'placate', 'rejuvinate']
#
# syns_messy = ['cornballs', 'slinging', 'slingin', 'airballs', 'MFers', 'bozo', 'pouding', 'gay-ass', 'cheesers', 'flippity', 'armor-piercing', 'watusi', 'knucklehead', 'baffoon', 'pansy-ass', 'goobers', 'quicksand', 'MFing', 'cowpie', 'pantload', 'knuckleheads', 'dumb-dumb', 'Andouille', 'squealin', 'duck', 'stroh', 'chucklehead', 'buckwild', 'hot-potato', 'ballz']
# ants_messy = ['tidy', 'tidiness', 'tidy.I', 'tidyness', 'tidy.', 'Tidiness', 'tidier', 'tidy.The', 'neatness', 'spotless', 'uncluttered', 'clean', 'spotlessly', 'Tidy', 'spick-and-span', 'cleanliness', 'neat', 'clutter-free', 'tidy-', 'orderliness', 'immaculately', 'spotlessness', 'spic-and-span', 'immaculate', 'houseproud', 'tidied', 'kempt', 'well-kept', 'tidily', 'orderly']
#
# syns_arrogant = ['importunities', 'ignobility', 'ill-nature', 'ill-humour', 'self-accusation', 'castigations', 'servility', 'timorousness', 'subserviency', 'contumely', 'truckling', 'revilings', 'querulousness', 'contemptuousness', 'subservience', 'indecorum', 'impertinences', 'intermeddling', 'Papism', 'self-abasement', 'repine', 'obtruding', 'calumniated', 'supineness', 'intermeddle', 'complaisance', 'calumniating', 'traducers', 'tenour', 'disoblige']
# ants_arrogant = ['cool', 'awesome', 'smart', 'amazing', 'coool', 'cooool', 'nice', 'neat', 'coooool', 'cooooool', 'coooooool', 'Cool', 'coolest', 'amazingly', 'kewl', 'fantastic', 'cool.A', 'neat-o', 'awsome', 'cooooooool', 'Awesome', 'cool.This', 'neato', 'cool.It', 'cool-', 'cool.The', 'amzing', 'cool-looking', 'AWESOME', 'cool.']
#
# syn_ant_lists = [
#     syns_dumb, ants_dumb,
#     syns_ugly, ants_ugly,
#     syns_lazy, ants_lazy,
#     syns_clueless, ants_clueless,
#     syns_slow, ants_slow,
#     syns_annoying, ants_annoying,
#     syns_messy, ants_messy,
#     syns_arrogant, ants_arrogant,
# ]
#
# for word_list in syn_ant_lists:
#     print(clean_syn_ant(word_list))
#
# print(clean_syn_ant(ants_arrogant))