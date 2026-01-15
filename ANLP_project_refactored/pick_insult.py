import random

def pick_insult(sorted_word_list, scores):
    # Add clause when empty
    return sorted_word_list[0]

def pick_eval_insult(words, scale, k=5):
    return random.shuffle(words[:k])
