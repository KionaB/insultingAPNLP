from tabulate import tabulate
from pick_insult import pick_eval_insult
import textwrap
import csv
import random

#TODO: criteria on perplexity / commoness --> r/Roastme. use n-gram and/or tf-idf
    # semantic distance
    # length of word (short words might hit harder?)
    # part of speech refining --> focus on nouns mostly or remove words that have -ness etc
    # ensure diversity, words that are very similar need to be removed (slow vs slow-down)
    
    # Human criteria on likert scale (1-5):
    # 1a. Relevance: slow as a snail rather than slow as an apple
    # 1b. Linguistic fit: Slow as a snail rather than slow as a deaccelartion
    # 2a. perceived severity / insult strength: Dumb as a dead body is harsh, while dumb as a carpet is very tame
    # 2b. Humor / cleverness: Dumb as a dead body is rather morbid, while dumb as a carpet is unintentionally funny
    # 2c. Concreteness / imagery: does the word evoke a clear mental image? dumb as carpet does, dumb as a ballast less so.
    # 3. Preference: Which do you like best from the list?
#TODO: Automatisch invullen en loggen voor evaluatie

fields = ["insult", "scale", "model", "word", "relevance", "Linguistic fit" "severity", "humor", "concrete"]

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

def run_evaluation(ins, insult_scale, syns, ants, PCA_method, worse_comparator_words, scores):
    with open ("evaluation_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
    
        eval_word_list = pick_eval_insult(worse_comparator_words, scores, 5)

        print('''
            Welcome! This is the Human criteria evaluation model.
            There are a couple questions for each word that must be ansower on a likert scale from 1-5.
            First, about how the word fits in the sentence as a whole:
                1a. Relevance: slow as a snail rather than slow as an apple
                1b. Linguistic fit: Slow as a snail rather than slow as a deaccelartion
            Second, the impact of the word itself:
                2a. perceived severity / insult strength: Dumb as a dead body is harsh, while dumb as a carpet is very tame
                2b. Humor / cleverness: Dumb as a dead body is rather morbid, while dumb as a carpet is unintentionally funny
                2c. Concreteness / imagery: does the word evoke a clear mental image? dumb as carpet does, dumb as a ballast less so.
            And finally, the ranking of all candidates:
                3. Preference: Which do you like best from the list overall?
            ''')

        for top_words in eval_word_list:
            # random.shuffle(top_words) # prevent order bias
            for word in top_words:
                print(f"\nInsult: {ins}")
                print(f"Word: {word}")
                relevance = int(input("Relevance (scale 1-5): "))
                linFit = int(input("Linguistic fit (scale 1-5): "))
                severity = int(input("Severity (scale 1-5): "))
                humor = int(input("Humor (scale 1-5): "))
                concrete = int(input("Concreteness (scale 1-5): "))

                writer.writerow({
                    "insult": ins,
                    "scale": insult_scale,
                    "model": PCA_method,
                    "word": word,
                    "Relevance": relevance,
                    "Linguistic fit": linFit,
                    "severity": severity,
                    "humor": humor,
                    "concrete": concrete
                })

        # rows.append([
        #     ins,
        #     insult_scale,
        #     wrap_cell(compact_list(syns, 5)),
        #     wrap_cell(compact_list(ants, 5)),
        #     wrap_cell(", ".join(eval_word_list))
        # ])
        # print(
        #     tabulate(
        #         rows,
        #         headers=["Original insult", "Semantic scale", "Synonyms", "Antonyms", "Top-5 worse words"],
        #         tablefmt="fancy_grid",
        #         colalign=("left", "left", "left", "left", "left")
        #     )
        # )    