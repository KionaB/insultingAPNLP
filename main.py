import logging
from template_processor import get_insult_from_template, comeback_builder_from_template
import worse_comparator_generator as worse_gen
import nltk

logger = logging.getLogger(__name__)
logging.basicConfig(filename='insult_generator.log', level=logging.INFO)

if __name__ == "__main__":
    nltk.download('wordnet',quiet=True)
    # prompt input
    # TODO input sanitation
    # TODO add escalating against self option for testing (report eval)

    # TODO add error handling stuff

    # TODO add choosing worse comparator method
    # mode = input("select a mode: "
    #              "\n [1] "
    #              "\n [2] "
    #              "\n [3] ")

    while True:
        first_insult = input("Insult me, I dare you "
                             "\nTemplates: "
                             "\n{[X] are/is as [Y] as a [Z]} "
                             "\n{[X] are/is a [Y]} "
                             "\nOR type \"exit\" to exit\n")
        if "exit" == first_insult.lower().strip():
            break
        print("Your insult:", first_insult)
        template, subject, insult_scale, comparator = get_insult_from_template(first_insult)
        worse_comparator = worse_gen.get_worse_comparator(comparator, insult_scale)
        comeback = comeback_builder_from_template(first_insult, template, subject, worse_comparator, insult_scale)
        print(comeback)
