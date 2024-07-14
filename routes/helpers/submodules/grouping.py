import difflib
from typing import List

"""
Up-to-date Grouping Code

Currently does not need modification

Has reasonable test coverage

Future Plans:
- Implement more advanced similarity algorithms
    - Only has string similarity for now
- Add support for grouping based on multiple criteria
- Improve performance by optimizing the grouping algorithm
"""

class SimilarityGrouper:
    def __init__(self, threshold=0.8):
        """
        Initialize the StringGrouper with a similarity threshold.
        :param threshold: float, the minimum similarity ratio for strings to be grouped together (default 0.8).
        """
        self.threshold = threshold

    def group_strings(self, input_strings):
        """
        Group input strings by similarity.
        :param input_strings: list of strings to be grouped.
        :return: list of lists, where each sublist contains similar strings.
        """
        groups = []
        for string in input_strings:
            # Check if the string fits into any existing group
            added_to_group = False
            for group in groups:
                if self._is_similar(string, group[0]):
                    group.append(string)
                    added_to_group = True
                    break

            # If the string doesn't fit into any existing group, create a new group
            if not added_to_group:
                groups.append([string])
        return groups

    def _is_similar(self, string1, string2):
        """
        Determine if two strings are similar based on the threshold.
        :param string1: string to compare.
        :param string2: string to compare.
        :return: bool, True if strings are similar, False otherwise.
        """
        ratio = difflib.SequenceMatcher(None, string1, string2).ratio()
        return ratio >= self.threshold


class StringGroupingInterface:
    """
    A wrapper class that provides an interface to group strings based on their similarity.

    This class acts as a facade for various string grouping algorithms, allowing the user to
    employ different algorithms interchangeably through a common interface. The actual grouping
    logic is delegated to the algorithm instance provided during initialization.
    """

    def __init__(self, algorithm: SimilarityGrouper):
        """
        Initializes the StringGroupingInterface with a specific string grouping algorithm.

        :param algorithm: An instance of a class that implements the SimilarityGrouper interface,
                          which must have a `group_strings` method. This method should take a list
                          of strings as input and return a list of lists, where each sublist contains
                          strings grouped based on their similarity.
        """
        self.algorithm = algorithm

    def group(self, strings: List[str]) -> List[List[str]]:
        """
        Groups a list of strings into sublists of similar strings using the provided algorithm.

        This method serves as a wrapper around the `group_strings` method of the algorithm
        provided during the initialization of this interface. It allows the user to easily
        group strings without directly interacting with the underlying algorithm.

        :param strings: A list of strings to be grouped based on their similarity.
        :return: A list of lists, where each sublist contains strings that are similar to each other,
                 as determined by the algorithm.
        """
        return self.algorithm.group_strings(strings)
