from tabulate import tabulate
from pick_insult import pick_eval_insult
import textwrap
import csv
import os
import re

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

fields = ["insult", "scale", "model", "word", "Relevance", "Linguistic_fit", "severity", "humor", "concrete"]

EVAL_INSULTS = [
    # "You are as dumb as a rock",
    "You are as ugly as a troll",
    "You are as lazy as a sloth",
    # "You are as clueless as a child",
    "You are as slow as a snail",
    "You are as annoying as a fly",
    "You are as messy as a pig",
    # "You are as arrogant as a king"
]

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

def get_next_eval_filename(model, prefix="evaluation_log"):
    """Creates filename if a new csv file must be added with icreased index"""
    pattern = re.compile(rf"{prefix}(\d+)_{re.escape(model)}\.csv")
    max_num = 0

    for fname in os.listdir("."):
        match = pattern.match(fname)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return f"{prefix}{max_num + 1}_{model}.csv"

def get_latest_eval_file(model, prefix="evaluation_log"):
    """Finds the most recent evalation CSV for a model
    This is important for resuming the latest evaluation instead of starting over"""
    pattern = re.compile(rf"{prefix}(\d+)_{re.escape(model)}\.csv")
    files = []

    for fname in os.listdir("."):
        match = pattern.match(fname)
        if match:
            files.append((int(match.group(1)), fname))
    
    if not files:
        return None

    return max(files, key=lambda x: x[0])[1]

def get_evaluated_insults(csv_file):
    """Determine which insults have already been evaluated in a given CSV file
    Returns the set of insults that have been covered in the csv file"""
    evaluated = set()

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["insult"]:
                evaluated.add(row["insult"])

    return evaluated

def get_eval_file_and_remaining_insults(model):
    latest_file = get_latest_eval_file(model)

    # No file yet -> start fresh
    if latest_file is None:
        new_file = get_next_eval_filename(model)
        return new_file, EVAL_INSULTS
    
    evaluated = get_evaluated_insults(latest_file)

    # evaluation complete? -> start fresh
    if set(EVAL_INSULTS).issubset(evaluated):
        new_file = get_next_eval_filename(model)
        return new_file, EVAL_INSULTS
    
    # Otherwise continue from left over insults
    remaining = [ins for ins in EVAL_INSULTS if ins not in evaluated]
    return latest_file, remaining


def run_evaluation(ins, insult_scale, model_name, worse_comparator_words, scores, filename):
    file_exists = os.path.exists(filename)

    with open (filename, "a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
    
        eval_word_list = pick_eval_insult(worse_comparator_words, scores, 5)
        
        print("\nCurrent insult: " + ins)
        print("\nCandidate words:")
        for i, word in enumerate(eval_word_list, start=1):
            print(f"{i}. {word}")

        for word in eval_word_list:
            print(f"Current word to evaluate: {word}")
            relevance = int(input("Relevance (scale 1-5): "))
            linFit = int(input("Linguistic fit (scale 1-5): "))
            severity = int(input("Severity (scale 1-5): "))
            humor = int(input("Humor (scale 1-5): "))
            concrete = int(input("Concreteness (scale 1-5): "))

            writer.writerow({
                "insult": ins,
                "scale": insult_scale,
                "model": model_name,
                "word": word,
                "Relevance": relevance,
                "Linguistic_fit": linFit,
                "severity": severity,
                "humor": humor,
                "concrete": concrete,
            })
            
        preference = int(input(
            f"\nWhich word do you prefer overall? (index 1 to {len(eval_word_list)}): "
        ))
        favorite_word = eval_word_list[preference - 1]

        writer.writerow({
            "insult": ins,
            "scale": insult_scale,
            "model": model_name,
            "word": favorite_word,
            "Relevance": "",
            "Linguistic_fit": "",
            "severity": "",
            "humor": "",
            "concrete": "",
        })
