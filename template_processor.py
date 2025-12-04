import logging
import re
logger = logging.getLogger(__name__)
def get_insult_from_template(first_insult:str):
    """Parses an input insult using one of the following templates:
     0: [X] are/is as [Y] as a [Z]
     1: [X] are/is a [Z]
     :arg str first_insult: An input insult to be parsed
     :returns int template: The detected template as numbered above
      str subject: the subject or target of the insult according to template
      str insult_scale (can be None): the scale on which the subject is compared
      str comparator: what the subject is compared to
     """
    #Set templates
    template = 1
    template_0 = r"( are as | is as )"
    zero_compiled = re.compile(template_0)
    template_1 = r"( are a | is a | are an | is an )"
    one_compiled = re.compile(template_1)
    insult_scale = None
    if zero_compiled.search(first_insult):
        template = 0
        pt1 = re.sub(zero_compiled, "|", first_insult)
        template_second = r"( as a | as an )"
        second_compiled = re.compile(template_second)
        variables = re.sub(second_compiled, "|", pt1).split("|")
        subject = variables[0]
        insult_scale = variables[1]
        comparator = variables[2]
    elif one_compiled.search(first_insult):
        variables = re.sub(one_compiled, "|", first_insult).split("|")
        subject = variables[0]
        comparator = variables[1]

    logger.info("Detected variables: "+str(variables))
    logger.info("Detected template: "+str(template))
    return template, subject, insult_scale, comparator


def get_worse_comparator(comparator:str, scale=None):
    worse_comparator = " "
    return worse_comparator


def comeback_builder_from_template(first_insult:str, template:int, subject:str, worse_comparator:str, insult_scale=" "):
    """Generates a return insult based on one of these templates:
     0: [X] are/is as [Y] as a [Z]
     1: [X] are/is a [Z]
     :arg str first_insult: The original insult to be parsed
      int template: The detected template as numbered above
      str subject: the subject or target of the insult according to template
      str worse_comparator: what the subject is compared to
      str insult_scale (can be None): the scale on which the subject is compared
      :returns str comeback: the constructed return insult
     """
    #Builds a comeback out of the given variables based on template first insult used

    comeback = "Comeback: " + subject
    if template == 0:
        if " is as " in first_insult:
            comeback = comeback + " is as " + insult_scale
        else:
            comeback = comeback + " are as " + insult_scale
        if worse_comparator[0] in ['a', 'e', 'i', 'o', 'u']:
            comeback = comeback + " as an " + worse_comparator
        else:
            comeback = comeback + " as a " + worse_comparator
    if template == 1:
        if " is " in first_insult:
            comeback = comeback + " is "
        else:
            comeback = comeback + " are "
        if worse_comparator[0] in ['a', 'e', 'i', 'o', 'u']:
            comeback = comeback + "an " + worse_comparator
        else:
            comeback = comeback + "a " + worse_comparator
    return comeback
