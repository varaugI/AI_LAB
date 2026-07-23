"""Evaluation metrics for complete OCR text."""


def levenshtein_distance(expected, predicted):
    previous = list(range(len(predicted) + 1))
    for row_index, expected_item in enumerate(expected, start=1):
        current = [row_index]
        for column_index, predicted_item in enumerate(predicted, start=1):
            substitution = previous[column_index - 1] + (expected_item != predicted_item)
            insertion = current[column_index - 1] + 1
            deletion = previous[column_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def character_error_rate(expected, predicted):
    if expected == "":
        return 0.0 if predicted == "" else 1.0
    return levenshtein_distance(expected, predicted) / len(expected)


def word_error_rate(expected, predicted):
    expected_words = expected.split()
    predicted_words = predicted.split()
    if not expected_words:
        return 0.0 if not predicted_words else 1.0
    return levenshtein_distance(expected_words, predicted_words) / len(expected_words)
