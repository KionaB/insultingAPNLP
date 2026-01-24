
def yn_input(prompt):
    while True:
        user_in = input(f"{prompt} (y/n or 'exit'): ").lower().strip()

        if user_in == "exit":
            return None
        elif user_in in ("y", "yes"):
            return True
        elif user_in in ("n", "no"):
            return False
        else:
            print("Invalid input, please enter y, n, or exit.")
    
def scale_gen_input():
    while True:
        choice = input(
            "Pick which scale generator to use:\n"
            "  1) WordNet\n"
            "  2) FastText\n"
        ).lower().strip()

        if choice == "exit":
            return None, None

        elif choice == "1":
            return "wordnet", "wordnet"

        elif choice == "2":
            extremes = yn_input("Use extremes?")
            if extremes:
                return "fasttext", "extremes"
            else:
                return "fasttext", "fasttext"
        else:
            print("Invalid input, please choose 1, 2, or exit.")
    
    
def generator_settings():
    NUM_ROUNDS = 5
    exit = False
    eval_answer = yn_input("Are you evaluating?")
    if eval_answer is None:
        return None, None, None, None, None, None, True # exit
    if eval_answer:
        return True, None, None, None, None, None, None # stop earlier
    
    self_battle_answer = yn_input("Should the generator battle itself?")
    if self_battle_answer is None:
        return None, None, None, None, None, None, True # exit
    elif self_battle_answer:
        while True:
            rounds = input("How many rounds do you want me to battle myself for?")
            try:
                NUM_ROUNDS = int(rounds)
                break
            except ValueError:
                print("Please enter a valid integer.")
    pca_answer = yn_input("Use Principle Component Analysis (PCA)?")
    if pca_answer is None:
        return None, None, None, None, None, None, True # exit
    if pca_answer:
        projection_model = 'PCA'
    else:
        projection_model = 'norm'
    vec_model, sys_ant_model = scale_gen_input()
    if vec_model is None:
        return None, None, None, None, None, None, True # exit
    
    return eval_answer, self_battle_answer, NUM_ROUNDS, vec_model, sys_ant_model, projection_model, exit