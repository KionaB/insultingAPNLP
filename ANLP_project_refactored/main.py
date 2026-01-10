from string import whitespace

print('file started')
import logging

from pick_insult import pick_insult, pick_eval_insult
from template_processor import get_insult_from_template, comeback_builder_from_template
from worse_comparator_generator import get_worse_comparator, get_multiple_anchor_comparator
from syn_ant_generation import *
from word_list_gen import *
from tabulate import tabulate
import nltk
import textwrap

print('file started2')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='insult_generator.log', level=logging.INFO)

#TODO: remove words ending on -ness and those that are similar to the insult, etc

#TODO: criteria on perplexity / commoness --> r/Roastme
#TODO: Automatisch invullen en loggen voor evaluatie
#TODO: runtime toevoegen

def compact_list(words, k=5):
    """Deduplicate and keep at most k words."""
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) == k:
            break
    return ", ".join(seen)

def wrap_cell(text, width=40):
    """Wrap long table cells."""
    return "\n".join(textwrap.wrap(text, width))


EVAL_INSULTS = [
    "You are as dumb as a rock",
    "You are as ugly as a troll",
    "You are as lazy as a sloth",
    "You are as clueless as a child",
    "You are as slow as a snail",
    "You are as annoying as a fly",
    "You are as messy as a pig",
    "You are as arrogant as a king"
]

self_battle = False  # Let the insult generator fight against itself
NUM_ROUNDS = 5      # Determine the amount of insults are generated during self battle
PCA_method = True   # Enable PCA for semantic scale ranking calculation
evaluation = True  # Turn on for evaluation mode to get top 5 words for different insults

def generate_comeback(insult,mode):
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

    pca_in = input("PCA? (y/n)").lower().strip(whitespace)
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


    if self_battle:
        print(
            "Insult me, I dare you "
            "\nTemplates: "
            "\n{[X] are/is as [Y] as a [Z]} "
            "\n{[X] are/is a [Y]} "
            "\nOR type \"exit\" to exit\n"
        )
        insult = input(
                "Write The first insult: "
            )
        for num in range(1, NUM_ROUNDS+1):
            comeback = generate_comeback(insult,mode)
            print(f"Round {num} comeback: {comeback}")
            insult = comeback
    elif evaluation:
        rows = []
        modes = ['wordnet', 'fasttext', 'extremes']
        print("Evaluating using modes: ",str(modes))
        print("Using insults: ",str(EVAL_INSULTS))
        for m in modes:
            for ins in EVAL_INSULTS:
                print(ins)
                template, subject, insult_scale, comparator = get_insult_from_template(ins)
                syns, ants, ants_found = get_scale_syns_and_opposites(insult_scale,m)
                if not ants_found:
                    print('No antonyms found for scale ' + str(insult_scale))
                words_for_comparator = get_close(comparator)
                worse_comparator_words, scores = get_worse_comparator(syns, ants, words_for_comparator, template, pca_method=PCA_method)
                eval_word_list = pick_eval_insult(worse_comparator_words, scores, 5)
                # rows.append([ins, insult_scale, ", ".join(syns), ", ".join(ants), ", ".join(eval_word_list)])
                rows.append([
                    ins,
                    insult_scale,
                    wrap_cell(compact_list(syns, 5)),
                    wrap_cell(compact_list(ants, 5)),
                    wrap_cell(", ".join(eval_word_list))
                ])
            print("Using mode: ", m)
            print(
                tabulate(
                    rows,
                    headers=["Original insult", "Semantic scale", "Synonyms", "Antonyms", "Top-5 worse words"],
                    tablefmt="fancy_grid",
                    colalign=("left", "left", "left", "left", "left")
                )
            )
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
                comeback = generate_comeback(insult,mode)
                print(comeback)
