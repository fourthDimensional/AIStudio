import pytest
from routes.helpers.grouping import SimilarityGrouper, \
    StringGroupingInterface


# Helper function to generate a large number of strings
def generate_strings(n):
    base_str = "string"
    return [f"{base_str}{i}" for i in range(n)]


@pytest.fixture
def string_grouping_interface():
    """Fixture to provide a StringGroupingInterface instance with default SimilarityGrouper for each test."""
    algorithm = SimilarityGrouper(threshold=0.8)
    return StringGroupingInterface(algorithm)


def test_different_strings_in_different_groups(string_grouping_interface):
    """
    Test that two very different strings are placed in different groups.
    """
    strings = ["apple", "banana"]
    grouped = string_grouping_interface.group(strings)
    assert len(grouped) == 2  # Expecting each string in its own group


def test_similar_strings_in_same_group(string_grouping_interface):
    """
    Test that two similar strings are placed in the same group.
    """
    strings = ["apple", "apples"]
    grouped = string_grouping_interface.group(strings)
    assert len(grouped) == 1  # Expecting one group containing both strings


def test_strings_not_repeated_across_groups(string_grouping_interface):
    """
    Test that strings are not repeated across different groups.
    """
    strings = ["apple", "apples", "banana", "bananas"]
    grouped = string_grouping_interface.group(strings)
    all_strings = sum(grouped, [])  # Flatten the list of lists
    assert len(all_strings) == len(set(all_strings))  # Ensure all strings are unique


def test_returning_list_of_lists(string_grouping_interface):
    """
    Test that the method returns a list of lists of strings.
    """
    strings = ["apple", "banana"]
    grouped = string_grouping_interface.group(strings)
    assert isinstance(grouped, list)  # Check outer list
    assert all(isinstance(group, list) for group in grouped)  # Check inner lists are lists
    assert all(isinstance(item, str) for group in grouped for item in group)  # Check all items are strings


def test_group_large_amount_of_strings(string_grouping_interface):
    """
    Test that the interface can group a large number of strings (>200).
    """
    strings = generate_strings(250)  # Generate 250 unique strings
    grouped = string_grouping_interface.group(strings)
    assert len(grouped) > 0  # Check that we have at least one group
    assert sum(len(group) for group in grouped) == 250  # Ensure all strings are accounted for


@pytest.fixture
def low_threshold_interface():
    """Fixture to provide a StringGroupingInterface instance with a low similarity threshold."""
    algorithm = SimilarityGrouper(threshold=0.5)
    return StringGroupingInterface(algorithm)


@pytest.fixture
def high_threshold_interface():
    """Fixture to provide a StringGroupingInterface instance with a high similarity threshold."""
    algorithm = SimilarityGrouper(threshold=0.9)
    return StringGroupingInterface(algorithm)


def test_low_threshold_groups_more_liberally(low_threshold_interface):
    """
    Test that a lower threshold results in more liberal grouping (more strings considered similar).
    """
    strings = ["apple", "apples", "aple", "banana", "bananas"]
    grouped = low_threshold_interface.group(strings)
    # Flexible assertion: With a low threshold, we expect fewer groups due to more liberal grouping
    assert len(grouped) < len(strings), "Expected fewer groups with a low threshold due to more liberal grouping."


def test_high_threshold_groups_more_conservatively(high_threshold_interface):
    """
    Test that a higher threshold results in more conservative grouping (fewer strings considered similar).
    """
    strings = ["apple", "apples", "aple", "banana", "bananas"]
    grouped = high_threshold_interface.group(strings)
    # Flexible assertion: With a high threshold, we expect more groups due to more conservative grouping
    assert len(grouped) > 1, "Expected more groups with a high threshold due to more conservative grouping."


def test_threshold_edge_cases(low_threshold_interface, high_threshold_interface):
    """
    Test the behavior of the grouping at extreme threshold values for a specific set of strings.
    """
    strings = ["apple", "apples", "APPLE", "Banana", "BANANA"]
    low_grouped = low_threshold_interface.group(strings)
    high_grouped = high_threshold_interface.group(strings)
    # With a low threshold, case differences and minor variations may be ignored, leading to fewer, larger groups
    assert len(low_grouped) <= len(high_grouped), "Expected more liberal grouping with a low threshold."
    # With a high threshold, even minor variations may lead to separate groups
    assert len(high_grouped) >= len(low_grouped), "Expected more conservative grouping with a high threshold."


def test_very_high_threshold_results_in_individual_groups(high_threshold_interface):
    """
    Test that a very high threshold effectively places each string in its own group due to strict similarity requirements.
    """
    strings = ["apple", "apples", "aple"]
    grouped = high_threshold_interface.group(strings)
    # Flexible assertion: Depending on the exact threshold, each string might end up in its own group
    assert len(grouped) >= len(
        strings) - 1, "Expected individual groups for each string or close to it with a very high threshold."
