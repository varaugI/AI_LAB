import random


def one_hot_encode(class_index, number_of_classes):
    if not 0 <= class_index < number_of_classes:
        raise ValueError("class_index is outside the class range.")
    encoded = [0.0 for _ in range(number_of_classes)]
    encoded[class_index] = 1.0
    return encoded


def train_test_split(inputs, targets, test_size=0.2, shuffle=True, seed=42):
    if not inputs or len(inputs) != len(targets):
        raise ValueError("inputs and targets must be non-empty and equally sized.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    indices = list(range(len(inputs)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    test_count = max(1, int(round(len(inputs) * test_size)))
    test_ids = set(indices[:test_count])

    train_inputs, train_targets = [], []
    test_inputs, test_targets = [], []
    for index in range(len(inputs)):
        if index in test_ids:
            test_inputs.append(inputs[index])
            test_targets.append(targets[index])
        else:
            train_inputs.append(inputs[index])
            train_targets.append(targets[index])

    return train_inputs, test_inputs, train_targets, test_targets
