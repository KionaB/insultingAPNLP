from string import whitespace
import logging
import timeit

from pick_insult import pick_insult
from template_processor import get_insult_from_template, comeback_builder_from_template
from worse_comparator_generator import get_worse_comparator
from syn_ant_generation import *
from word_list_gen import *
from Evaluation import *
import nltk
from WordCleanUp import clean_syn_ant

logger = logging.getLogger(__name__)
logging.basicConfig(filename='insult_generator.log', level=logging.INFO)

#TODO: remove duplicate words                               Timon

#TODO: Create statistics for the final evaluation csv file  Nathan
#TODO: for in report, add syn ants example list, everything for a couple example insults for in the appendices. 
#TODO: result must pick random synonym to make it more 'creative'

#TODO: mooie plaatjes van hirarchy ?                        Kiona

def generate_comeback(insult,mode):
    """generate a comeback for any given insult"""
    template, subject, insult_scale, comparator = get_insult_from_template(insult)
    if insult_scale is None:
        insult_scale = "terrible"
    syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale, mode) # choose between 'wordnet', 'fasttext' and 'extremes', but extremes does not work with worse_comparator yet
    syns, ants = clean_syn_ant(syns), clean_syn_ant(ants) # Clean the synonyms and antonyms
    if not ants_found:
        logger.warning('No antonyms found for scale ' + str(insult_scale))
    words_for_comparator = get_close(comparator)
    print('synonyms: ', syns)
    print('antonyms: ', ants)
    worse_comparator_words, scores = get_worse_comparator(syns, ants, insult_scale, words_for_comparator, 
                                                            pca_method=PCA_method, mid_adjust=True, vec_model=mode, 
                                                            similarity_threshold = 2, max_words = None) 
    worse_comparator = pick_insult(worse_comparator_words)
    logger.info("Increased step comparator: " + worse_comparator)
    comeback = comeback_builder_from_template(insult, template, subject, worse_comparator, insult_scale)
    return comeback

if __name__ == "__main__":
    nltk.download('wordnet',quiet=True)
    battle_self_in = input("Do you want to watch the computer battle itself? (y/n)").lower().strip(whitespace)
    if battle_self_in == "y" or battle_self_in == "yes":
        print('battling myself')
        self_battle = True
        while True:
            user_input = input(
                "How many rounds do you want me to battle myself for?"
            )
            try:
                NUM_ROUNDS = int(user_input)
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        self_battle = False

    pca_in = input("Use PCA? (y/n)").lower().strip(whitespace)
    if pca_in == "y" or pca_in == "yes":
        print('pca on')
        PCA_method = True
        model_name = 'pca'
    else:
        PCA_method = False
        model_name = 'norm'

    eval_in = input("Are you evaluating? (y/n)").lower().strip(whitespace)
    if eval_in == "y" or eval_in == "yes":
        print('evaluating')
        evaluation = True
    else:
        evaluation = False
        while True:
            mode_num = input("Please select a mode: 1 'wordnet', 2 'fasttext', or 3 'extremes'")
            if mode_num.lower().strip(whitespace) == "1":
                mode = 'wordnet'
                break
            elif mode_num.lower().strip(whitespace) == "2":
                mode = 'fasttext'
                break
            elif mode_num.lower().strip(whitespace) == "3":
                mode = 'extremes'
                break
            elif mode_num.lower().strip(whitespace) == "exit":
                mode_num = 'exit'
                self_battle = False
                evaluation = False
                break
            else:
                print("your mode was: ", mode_num, ".Mode must be '1', '2' or '3'")
    # prompt input


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

    if self_battle:
        insult = input(
                "Write The first insult: "
            )
        for num in range(1, NUM_ROUNDS+1):
            comeback = generate_comeback(insult,mode)
            print(f"Round {num} comeback: {comeback}")
            insult = comeback
    elif evaluation:
        #TODO: Loop only over wordnet+wordnet, fasttext+fasttext, and fasttext+extremes. Furthermore PCA on/off, midadjust on/off?
        modes = ['wordnet', 'extremes']
        print("Evaluating using modes: ",str(modes))
        print("Using insults: ",str(EVAL_INSULTS))
        for m in modes:
            filename, remaining_insults = get_eval_file_and_remaining_insults(model_name, m)
            print("Evaluating mode: ", m)
            for ins in remaining_insults:
                template, subject, insult_scale, comparator = get_insult_from_template(ins)
                syns_ants_time_start = timeit.default_timer()
                syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale, 'wordnet')
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
                                                            pca_method=PCA_method, mid_adjust=True, vec_model=m, 
                                                            similarity_threshold = 2, max_words = None)
                worse_comp_end_time = timeit.default_timer()
                logger.info('time to get alternative comparator words: ' + str(worse_comp_start_time-alt_words_start))
                logger.info('time to get worse comparator scores: ' + str(worse_comp_end_time-worse_comp_start_time))
                print(f"Using evaluation file: {filename}")
                print(f"Remaining insults left to evaluate: {remaining_insults}")
                syn_ant_speed = syns_ants_time_end-syns_ants_time_start
                clean_syn_ant_speed = clean_time_end-syns_ants_time_end
                alt_words_speed = worse_comp_start_time-alt_words_start
                worse_comparator_speed = worse_comp_end_time-worse_comp_start_time
                completed = run_evaluation(ins, insult_scale, ants_found, model_name, worse_comparator_words, filename,syn_ant_speed,clean_syn_ant_speed,alt_words_speed,worse_comparator_speed)
                if not completed:
                    print("Stopped evaluation for mode: "+str(m))
                    break

    else: 
        if mode_num !='exit':
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
            comeback = generate_comeback(insult, mode)
            print(comeback)
