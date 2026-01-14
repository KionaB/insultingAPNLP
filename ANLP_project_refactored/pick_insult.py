import random

def pick_insult(sorted_word_list, scores):
    # Add clause when empty
    return sorted_word_list[0]

def pick_eval_insult(words, scale, k=5):
    unique_words = []

    for word in words:
        if word not in unique_words:
            unique_words.append(word)

        if len(unique_words) == k:
            break
    random.shuffle(unique_words)
    return unique_words
