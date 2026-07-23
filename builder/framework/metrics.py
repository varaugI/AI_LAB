def argmax(values):
    if not values:
        raise ValueError("values cannot be empty.")
    best_index = 0
    for index in range(1, len(values)):
        if values[index] > values[best_index]:
            best_index = index
    return best_index


def classification_accuracy(predictions, targets):
    if not predictions or len(predictions) != len(targets):
        raise ValueError("predictions and targets must be non-empty and equally sized.")

    correct = 0
    for prediction, target in zip(predictions, targets):
        if argmax(prediction) == argmax(target):
            correct += 1
    return correct / len(predictions)


def binary_accuracy(predictions, targets, threshold=0.5):
    if not predictions or len(predictions) != len(targets):
        raise ValueError("predictions and targets must be non-empty and equally sized.")

    correct = 0
    total = 0
    for prediction_row, target_row in zip(predictions, targets):
        if len(prediction_row) != len(target_row):
            raise ValueError("Prediction and target shapes do not match.")
        for prediction, target in zip(prediction_row, target_row):
            correct += (prediction >= threshold) == (target >= threshold)
            total += 1
    return correct / total


def confusion_matrix(predictions, targets, number_of_classes=None):
    predicted_classes = [argmax(row) for row in predictions]
    target_classes = [argmax(row) for row in targets]

    if number_of_classes is None:
        number_of_classes = max(predicted_classes + target_classes) + 1

    matrix = [[0 for _ in range(number_of_classes)] for _ in range(number_of_classes)]
    for actual, predicted in zip(target_classes, predicted_classes):
        matrix[actual][predicted] += 1
    return matrix
