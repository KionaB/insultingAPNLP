import random

def pick_insult(sorted_word_list):
    # Add clause when empty
    return sorted_word_list[0]

def pick_eval_insult(words, k=5):
    #return random.shuffle(words[:k])  #random werkt niet?????
    return words[:k]
