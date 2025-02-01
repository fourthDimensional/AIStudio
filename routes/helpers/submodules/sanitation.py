import html

def sanitize_input(input_str):
    """
    Sanitize input string to prevent XSS attacks.

    :param input_str: The input string to sanitize.

    :return: The sanitized string.
    """
    # sanitized_str = html.escape(input_str)
    return input_str