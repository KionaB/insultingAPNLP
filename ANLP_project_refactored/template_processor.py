import logging
import re
import random

logger = logging.getLogger(__name__)


def get_insult_from_template(first_insult: str):
    """Parses an input insult using one of the following templates:
     0: [X] are/is as [Y] as a [Z]
     1: [X] are/is a [Z]
     :arg str first_insult: An input insult to be parsed
     :returns int template: The detected template as numbered above
      str subject: the subject or target of the insult according to template
      str insult_scale (can be None): the scale on which the subject is compared
      str comparator: what the subject is compared to
     """
    # Set templates
    zero_compiled = re.compile(r"^(?P<subject>.+?) (are|is) as (?P<insult_scale>.+?) as (a|an) (?P<comparator>.+)$", re.IGNORECASE)
    one_compiled = re.compile(r"^(?P<subject>.+?) (are|is) (a|an) (?P<comparator>.+)$", re.IGNORECASE)
    template = None
    insult_scale = None
    subject = ""
    comparator = ""
    match0 = zero_compiled.match(first_insult)
    match1 = one_compiled.match(first_insult)
    if zero_compiled.search(first_insult):
        template = 0
        subject = match0.group("subject").strip()
        insult_scale = match0.group("insult_scale").strip()
        comparator = match0.group("comparator").strip()
    elif one_compiled.search(first_insult):
        template = 1
        subject = match1.group("subject").strip()
        comparator = match1.group("comparator").strip()

    logger.info(f"Detected variables: template = {template}, subject = {subject}, scale = {insult_scale}, comparator = {comparator}")
    return template, subject, insult_scale, comparator


def comeback_builder_from_template(syns, first_insult: str, template: int, subject: str, worse_comparator: str,
                                   insult_scale=" "):
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
    insult_scale = random.choice(syns)
    comeback = subject
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
