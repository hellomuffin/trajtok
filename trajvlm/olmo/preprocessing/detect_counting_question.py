import re

non_countable_quantities = [
    # time
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "[a-z]*seconds",

    # length/area
    "(tera|giga|mega|deci|kilo||micro|centi|milli|nano|pico|deca)meters",
    "meters",
    "metres",  # mispelt meters
    "acres",
    "leagues",
    "fathoms",
    "nautical miles",
    "hectares",
    "(square |SQ )?inches",
    "(square |SQ )feet",  # Just feet can be a false positive
    "(square |SQ )?ft",
    "(square |SQ )?miles",
    "(square | SQ)?yards",
    "passing yards",

    # currency
    "dollars",
    "cents",
    "pounds",
    "euros",

    # speed
    "seed",
    "mph",
    "kph",

    # Comparisons "how many more..."
    "more",
    "fewer",
    "less",

    "likes",  # almost always from a screenshot

    # volume
    "cubic",
    "gallons",
    "quarts",
    "pints",
    "fluid ounces",
    "[a-z]*liters",
    # ambiguous and probably more often used as an object then a volume
    # cup
    # tablespoons
    # teaspoons

    # weight
    "weight",
    "[a-z]*grams",
    "pounds",
    "tons",
    "ounces",

    "ways", "different ways",

    # other
    "degrees", "calories",
    "hertz", "horsepower", "[a-z]*bytes",
    "psi", "atmospheres", "[a-z]*watts",
]
non_countable_re_str = "|".join(non_countable_quantities)
non_countable_end_re_str = "|".join(non_countable_quantities + ["money", "the"])

counting_patterns = [
    f'how ?many (?!{non_countable_re_str})',
    r'(?<!do not )(count|tally) (all|every|each|the) ',
    "(there are|a total of) _{3,4}",
    f"(what|(what's|what (is|was|were)|states?|indicates?) the( exact| precise)?) (total count|count|total|total number|number|num|total amount|amount) of (?!{non_countable_end_re_str})",
]
count_any = re.compile("^(?!approximately).*(\\b|^|\n)(?P<all>" + "|".join(counting_patterns) + ")\\b.*", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def is_pixmo_point_and_count_question(question):
    """
    Could this question be counting question that the model should use pointing for?

    This check is conservative, so it will have a high recall but low precision
    """
    return bool(count_any.fullmatch(question))
