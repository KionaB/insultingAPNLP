import logging
import timeit
import nltk
import fasttext
import fasttext.util

from pick_insult import pick_insult
from template_processor import get_insult_from_template, comeback_builder_from_template
from worse_comparator_generator import get_worse_comparator
from setup import generator_settings
from syn_ant_generation import get_scale_syns_and_opposites
from word_list_gen import get_close
from Evaluation import *
from WordCleanUp import clean_syn_ant

logger = logging.getLogger(__name__)
logging.basicConfig(filename='insult_generator.log', level=logging.INFO)
stop_evaluation = False

fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.en.300.bin')

eval_settings = ( #eval settings looks like [vec_model, sys_ant_model, projection_model]
    ('wordnet',  'wordnet', 'norm'),    # 1. hesitancy  2. reluctance   3. disinclination   4. overeating   5. avaritia
    ('fasttext', 'fasttext', 'norm'),   # 1. leviathan  2. trolling     3. Typhon           4. hellhound    5. Typhoeus
    ('fasttext', 'extremes', 'norm'),   # 1. trolling   2. wolfman      3. Typhon           4. lycanthrope  5. leviathan
    ('wordnet',  'wordnet', 'PCA'),     # 1. salamander 2. hellhound    3. mantichora       4. werewolf     5. leviathan
    ('fasttext', 'fasttext', 'PCA'),    # 1. firedrake  2. partsong     3. fly              4. Geryon       5. fisherman's lure
    ('fasttext', 'extremes', 'PCA'),    # 1. partsong   2. fly-fishing  3. firedrake        4. roc          5. Sphinx
)

#TODO: Create statistics for the final evaluation csv file  Nathan
#TODO: result must pick random synonym to make it more 'creative'

def generate_comeback(insult, vec_model, sys_ant_model, projection_model, fasttext_model):
    """generate a comeback for any given insult"""
    template, subject, insult_scale, comparator = get_insult_from_template(insult)
    if insult_scale is None:
        insult_scale = "terrible"
    syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale, fasttext_model, sys_ant_model) # choose between 'wordnet', 'fasttext' and 'extremes', but extremes does not work with worse_comparator yet
    syns, ants = clean_syn_ant(syns), clean_syn_ant(ants) # Clean the synonyms and antonyms
    if not ants_found:
        logger.warning('No antonyms found for scale ' + str(insult_scale))
    words_for_comparator = get_close(comparator)
    print('synonyms: ', syns)
    print('antonyms: ', ants)
    worse_comparator_words, scores = get_worse_comparator(syns, ants, insult_scale, words_for_comparator, 
                                                            projection_model=projection_model, mid_adjust=True, vec_model=vec_model, 
                                                            similarity_threshold = 2) 
    worse_comparator = pick_insult(worse_comparator_words)
    logger.info("Increased step comparator: " + worse_comparator)
    comeback = comeback_builder_from_template(syns, insult, template, subject, worse_comparator, insult_scale)
    return comeback

if __name__ == "__main__":
    nltk.download('wordnet',quiet=True)

    evaluation, self_battle, NUM_ROUNDS, vec_model, sys_ant_model, projection_model, exit= generator_settings()

    if evaluation:
        print('''
Welcome! This is the Human criteria evaluation model.
There are a couple questions for each word that must be answered on a likert scale from 1-5.
First, about how the word fits in the sentence as a whole:
    1a. Relevance: slow as a snail rather than slow as an apple
    1b. Linguistic fit: Slow as a snail rather than slow as a deaccelartion
Second, the impact of the word itself:
    2a. perceived severity / insult strength: Dumb as a dead body is harsh, while dumb as a carpet is very tame
    2b. Humor / cleverness: Dumb as a dead body is rather morbid, while dumb as a carpet is unintentionally funny
    2c. Concreteness / imagery: does the word evoke a clear mental image? dumb as carpet does, dumb as a ballast less so.
And finally, choosing which word ranked best overall:
    3. Preference: Which do you like best from the list overall?
            ''')
        print("Using insults: ",str(EVAL_INSULTS))
        for vec_model, sys_ant_model, projection_model in eval_settings:
            filename, remaining_insults = get_eval_file_and_remaining_insults(vec_model, sys_ant_model, projection_model)
            print(f"Evaluating mode: {vec_model}, {sys_ant_model}, {projection_model}")
            for ins in remaining_insults:
                template, subject, insult_scale, comparator = get_insult_from_template(ins)
                syns_ants_time_start = timeit.default_timer()
                syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale, fasttext_model, sys_ant_model)
                syns_ants_time_end = timeit.default_timer()
                syns, ants = clean_syn_ant(syns), clean_syn_ant(ants)
                clean_time_end = timeit.default_timer()
                logger.info('time to get syns and ants'+str(syns_ants_time_end-syns_ants_time_start))
                logger.info('time to clean syns and ants'+str(clean_time_end-syns_ants_time_end))
                print('synonyms: ', syns)
                print('antonyms: ', ants)
                if insult_scale is None:
                    insult_scale = "terrible"
                if not ants_found:
                    logger.warning('No antonyms found for scale ' + str(insult_scale))
                alt_words_start = timeit.default_timer()
                words_for_comparator = get_close(comparator)
                worse_comp_start_time = timeit.default_timer()
                worse_comparator_words, scores = get_worse_comparator(syns, ants, insult_scale, words_for_comparator, 
                                                            projection_model = projection_model, mid_adjust=True, vec_model = vec_model, 
                                                            similarity_threshold = 2)
                worse_comp_end_time = timeit.default_timer()
                logger.info('time to get alternative comparator words: ' + str(worse_comp_start_time-alt_words_start))
                logger.info('time to get worse comparator scores: ' + str(worse_comp_end_time-worse_comp_start_time))
                print(f"Using evaluation file: {filename}")
                print(f"Remaining insults left to evaluate: {remaining_insults}")
                syn_ant_speed = syns_ants_time_end-syns_ants_time_start
                clean_syn_ant_speed = clean_time_end-syns_ants_time_end
                alt_words_speed = worse_comp_start_time-alt_words_start
                worse_comparator_speed = worse_comp_end_time-worse_comp_start_time
                completed = run_evaluation(ins, insult_scale, ants_found, vec_model, sys_ant_model, projection_model, worse_comparator_words, filename,syn_ant_speed,clean_syn_ant_speed,alt_words_speed,worse_comparator_speed)
                if not completed:
                    print(f"Stopped evaluation for mode: {vec_model}, {sys_ant_model}, {projection_model}")
                    stop_evaluation = True
                    break
            if stop_evaluation:
                break
    elif self_battle:
        insult = input(
                "Write The first insult: "
                "\nTemplates: "
                "\n{[X] are/is as [Y] as a [Z]} "
                "\n{[X] are/is a [Y]} "
                "\nOR type \"exit\" to exit\n"
            )
        for num in range(1, NUM_ROUNDS+1):
            comeback = generate_comeback(insult,vec_model, sys_ant_model, projection_model, fasttext_model)
            print(f"Round {num} comeback: {comeback}")
            insult = comeback
    elif not exit:
            print(
                "Insult me, I dare you "
                "\nTemplates: "
                "\n{[X] are/is as [Y] as a [Z]} "
                "\n{[X] are/is a [Y]} "
                "\nOR type \"exit\" to exit\n"
            )
            while True:
                insult = input(
                    "Write an insult: "
                )
                if insult.lower().strip() == "exit":
                    break
                comeback = generate_comeback(insult, vec_model, sys_ant_model, projection_model, fasttext_model)
                print(comeback)
