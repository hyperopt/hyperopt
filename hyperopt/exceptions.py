"""
"""

class BadSearchSpace(Exception):
    """Something is wrong in the description of the search space"""


class DuplicateLabel(BadSearchSpace):
    """A search space included a duplicate label """


class InvalidTrial(ValueError):
    """Non trial-like object used as Trial"""


