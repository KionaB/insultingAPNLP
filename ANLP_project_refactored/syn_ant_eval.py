from syn_ant_generation import get_scale_syns_and_opposites
from WordCleanUp import clean_syn_ant
import time
import fasttext

fasttext_model = fasttext.load_model('cc.en.300.bin')
scales = ['ugly', 'lazy', 'slow', 'annoying', 'messy']
modes = ['wordnet', 'fasttext', 'extremes']

for mode in modes:
    start_time = time.time()
    print('')
    print('mode: ', mode)
    for scale in scales:
        print('scale: ', scale)

        synonyms, antonyms, ants_found = get_scale_syns_and_opposites(scale, fasttext_model, mode=mode)
        print('These are dirty syns: ', synonyms)
        print('These are dirty ants: ', antonyms)
        synonyms, antonyms = clean_syn_ant(synonyms), clean_syn_ant(antonyms)
        print('syns: ', ', '.join(synonyms))
        if not ants_found:
            print('We used emergency antonyms')
        print('ants: ', ', '.join(antonyms))
    print('final time: ', time.time() - start_time)
