print('file started')
import logging

from pick_insult import pick_insult
from template_processor import get_insult_from_template, comeback_builder_from_template
from worse_comparator_generator import get_worse_comparator, get_multiple_anchor_comparator
from syn_ant_generation import *
from word_list_gen import *
import nltk

print('file started2')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='insult_generator.log', level=logging.INFO)


self_battle = False  # Let the insult generator fight against itself
NUM_ROUNDS = 5      # Determine the amount of insults are generated during self battle
PCA_method = False   # Enable PCA for semantic scale ranking calculation

def generate_comeback(insult):
    """generate a comeback for any given insult"""
    template, subject, insult_scale, comparator = get_insult_from_template(insult)
    if insult_scale is None:
        insult_scale = "terrible"
    syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale, 'fasttext') # choose between 'wordnet', 'fasttext' and 'extremes', but extremes does not work with worse_comparator yet
    if not ants_found:
        logger.warn('No antonyms found for scale ' + str(insult_scale))
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
    # prompt input
    # TODO input sanitation
    # TODO add error handling stuff
    print(
                "Insult me, I dare you "
                "\nTemplates: "
                "\n{[X] are/is as [Y] as a [Z]} "
                "\n{[X] are/is a [Y]} "
                "\nOR type \"exit\" to exit\n"
                )
    
    # print(get_multiple_anchor_comparator("donkey", "dumb", method=True))  # PCA
    # print(get_multiple_anchor_comparator("donkey", "dumb", method=False)) # Multi-anchor

    if self_battle:
        insult = input(
                "Write an insult: "
            )
        for num in range(1, NUM_ROUNDS+1):
            comeback = generate_comeback(insult)
            print(f"Round {num} comeback: {comeback}")
            insult = comeback
    else:
        while True:
            insult = input(
                "Write an insult: "
            )
            if insult.lower().strip() == "exit":
                break
            comeback = generate_comeback(insult)
            print(comeback)
