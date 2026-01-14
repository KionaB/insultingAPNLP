from string import whitespace

print('file started')
import logging

from pick_insult import pick_insult
from template_processor import get_insult_from_template, comeback_builder_from_template
from worse_comparator_generator import get_worse_comparator, get_multiple_anchor_comparator
from syn_ant_generation import *
from word_list_gen import *
from Evaluation import *
import nltk

print('file started2')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='insult_generator.log', level=logging.INFO)

#TODO: remove duplicate words                               Timon

#TODO: remove those that are similar to the insult scale    Nathan
#TODO: add insult reminder during eval                      
#TODO: change evaluation criterea
#TODO: Create statistics for the final evaluation csv file
#TODO: Add check that 5 words are present in csv file per insult, otherwise start insult over
#TODO: Add comparator word in the evaluation csv list
#TODO: fix that pca selection happens through user input and mode selection (and the other stuff too)
#TODO: for in report, add syn ants example list, everything for a couple example insults for in the appendices. 

#TODO: runtime toevoegen                                    Kiona
#TODO: implement mode changes
#TODO: remove words ending on -ness
#TODO: mooie plaatjes van hirarchy ?




self_battle = False  # Let the insult generator fight against itself
NUM_ROUNDS = 5      # Determine the amount of insults are generated during self battle
PCA_method = True   # Enable PCA for semantic scale ranking calculation
evaluation = True  # Turn on for evaluation mode to get top 5 words for different insults

if PCA_method:
    model_name = 'pca'
else:
    model_name = 'norm'

def generate_comeback(insult):
    """generate a comeback for any given insult"""
    template, subject, insult_scale, comparator = get_insult_from_template(insult)
    if insult_scale is None:
        insult_scale = "terrible"
    syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale, mode) # choose between 'wordnet', 'fasttext' and 'extremes', but extremes does not work with worse_comparator yet
    if not ants_found:
        logger.warning('No antonyms found for scale ' + str(insult_scale))
    words_for_comparator = get_close(comparator)
    worse_comparator_words, scores = get_worse_comparator(syns, ants, words_for_comparator, template, pca_method=PCA_method)
    worse_comparator = pick_insult(worse_comparator_words, scores)
    logger.info("Increased step comparator: " + worse_comparator)
    comeback = comeback_builder_from_template(insult, template, subject, worse_comparator, insult_scale)
    return comeback

if __name__ == "__main__":
    print('start download')
    nltk.download('wordnet',quiet=True)
    print('end download')

    # TODO input sanitation
    # TODO add error handling stuff
    #Mode selection stuff
    battle_self_in = input("Do you want to watch the computer battle itself? (y/n)").lower().strip(whitespace)
    if battle_self_in == "y" or battle_self_in == "yes":
        print('battling myself')
        self_battle = True
    else:
        self_battle = False

    pca_in = input("Use PCA? (y/n)").lower().strip(whitespace)
    if pca_in == "y" or pca_in == "yes":
        print('pca on')
        PCA_method = True
    else:
        PCA_method = False

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
    else: 
        print(
            "Insult me, I dare you "
            "\nTemplates: "
            "\n{[X] are/is as [Y] as a [Z]} "
            "\n{[X] are/is a [Y]} "
            "\nOR type \"exit\" to exit\n"
        )

    if self_battle:
        insult = input(
                "Write The first insult: "
            )
        for num in range(1, NUM_ROUNDS+1):
            comeback = generate_comeback(insult,mode)
            print(f"Round {num} comeback: {comeback}")
            insult = comeback
    elif evaluation:
        filename, remaining_insults = get_eval_file_and_remaining_insults(model_name)
        modes = ['wordnet', 'fasttext', 'extremes']
        print("Evaluating using modes: ",str(modes))
        print("Using insults: ",str(EVAL_INSULTS))
        #for m in modes:
        for ins in remaining_insults:
            template, subject, insult_scale, comparator = get_insult_from_template(ins)
            if insult_scale is None:
                insult_scale = "terrible"
            syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale)
            if not ants_found:
                logger.warning('No antonyms found for scale ' + str(insult_scale))
            words_for_comparator = get_close(comparator)
            worse_comparator_words, scores = get_worse_comparator(syns, ants, words_for_comparator, template, pca_method=PCA_method)
            print(f"Using evaluation file: {filename}")
            print(f"Remaining insults left to evaluate: {remaining_insults}")
            run_evaluation(ins, insult_scale, model_name, worse_comparator_words, scores, filename)

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
