import logging
from template_processor import get_insult_from_template, get_worse_comparator, comeback_builder_from_template

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logging.basicConfig(filename='insult_generator.log', level=logging.INFO)
    # prompt input
    #TODO input sanitation
    first_insult = input("Insult me, I dare you \nTemplates: \n{[X] are/is as [Y] as a [Z]} \n{[X] are/is a [Y]} \n")
    print("Your insult:", first_insult)
    template, subject, insult_scale, comparator = get_insult_from_template(first_insult)
    worse_comparator = get_worse_comparator(comparator, insult_scale)
    logger.info("Increased step comparator: "+worse_comparator)
    comeback= comeback_builder_from_template(first_insult, template, subject, worse_comparator, insult_scale)
    print(comeback)