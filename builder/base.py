import json
import math
import random


# =========================================================
# MATRIX FUNCTIONS
# =========================================================

def matrix_shape(matrix):
    return len(matrix), len(matrix[0])


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows, cols, minimum=-1.0, maximum=1.0):
    return [
        [
            random.uniform(minimum, maximum)
            for _ in range(cols)
        ]
        for _ in range(rows)
    ]


def matmul(A, B):
    if not A or not B:
        raise ValueError("Matrices cannot be empty.")

    if len(A[0]) != len(B):
        raise ValueError(
            f"Cannot multiply matrices with shapes "
            f"{len(A)}x{len(A[0])} and "
            f"{len(B)}x{len(B[0])}."
        )

    rows = len(A)
    cols = len(B[0])
    inner = len(B)

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            total = 0.0

            for k in range(inner):
                total += A[i][k] * B[k][j]

            result[i][j] = total

    return result


def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    result = zeros(cols, rows)

    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]

    return result


def add(A, B):
    rows_a, cols_a = matrix_shape(A)
    rows_b, cols_b = matrix_shape(B)

    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError("Matrices must have the same shape.")

    result = zeros(rows_a, cols_a)

    for i in range(rows_a):
        for j in range(cols_a):
            result[i][j] = A[i][j] + B[i][j]

    return result


def subtract(A, B):
    rows_a, cols_a = matrix_shape(A)
    rows_b, cols_b = matrix_shape(B)

    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError("Matrices must have the same shape.")

    result = zeros(rows_a, cols_a)

    for i in range(rows_a):
        for j in range(cols_a):
            result[i][j] = A[i][j] - B[i][j]

    return result


def multiply_elementwise(A, B):
    rows_a, cols_a = matrix_shape(A)
    rows_b, cols_b = matrix_shape(B)

    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError("Matrices must have the same shape.")

    result = zeros(rows_a, cols_a)

    for i in range(rows_a):
        for j in range(cols_a):
            result[i][j] = A[i][j] * B[i][j]

    return result


def scalar_multiply(matrix, scalar):
    rows, cols = matrix_shape(matrix)

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[i][j] * scalar

    return result


def apply_function(matrix, function):
    rows, cols = matrix_shape(matrix)

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = function(matrix[i][j])

    return result


def add_bias(matrix, bias):
    """
    Adds one bias row to every row in the matrix.

    matrix:
        batch_size x neurons

    bias:
        1 x neurons
    """

    rows, cols = matrix_shape(matrix)

    if len(bias) != 1 or len(bias[0]) != cols:
        raise ValueError("Bias shape does not match matrix.")

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[i][j] + bias[0][j]

    return result


def sum_rows(matrix):
    """
    Converts:

    [
        [1, 2],
        [3, 4]
    ]

    into:

    [
        [4, 6]
    ]
    """

    rows, cols = matrix_shape(matrix)

    result = zeros(1, cols)

    for i in range(rows):
        for j in range(cols):
            result[0][j] += matrix[i][j]

    return result


# =========================================================
# ACTIVATION FUNCTIONS
# =========================================================

def sigmoid_value(value):
    value = max(min(value, 500), -500)

    return 1.0 / (1.0 + math.exp(-value))


def sigmoid(matrix):
    return apply_function(matrix, sigmoid_value)


def sigmoid_derivative_from_output(output):
    """
    If output = sigmoid(x), then:

    sigmoid derivative = output * (1 - output)
    """

    rows, cols = matrix_shape(output)

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            value = output[i][j]
            result[i][j] = value * (1.0 - value)

    return result


def relu_value(value):
    return max(0.0, value)


def relu(matrix):
    return apply_function(matrix, relu_value)


def relu_derivative(matrix):
    rows, cols = matrix_shape(matrix)

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = 1.0 if matrix[i][j] > 0 else 0.0

    return result


# =========================================================
# LOSS FUNCTIONS
# =========================================================

def mean_squared_error(predictions, targets):
    rows, cols = matrix_shape(predictions)

    total_error = 0.0
    number_of_values = rows * cols

    for i in range(rows):
        for j in range(cols):
            difference = predictions[i][j] - targets[i][j]
            total_error += difference ** 2

    return total_error / number_of_values


def mean_squared_error_derivative(predictions, targets):
    rows, cols = matrix_shape(predictions)

    number_of_values = rows * cols

    result = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = (
                2.0
                * (predictions[i][j] - targets[i][j])
                / number_of_values
            )

    return result


# =========================================================
# DENSE LAYER
# =========================================================

class DenseLayer:
    def __init__(self, input_size, output_size):
        scale = math.sqrt(2.0 / input_size)

        self.weights = random_matrix(
            input_size,
            output_size,
            -scale,
            scale,
        )

        self.biases = zeros(1, output_size)

        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs

        weighted_values = matmul(inputs, self.weights)

        self.output = add_bias(
            weighted_values,
            self.biases,
        )

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        output_gradient tells this layer how much the loss changes
        when the layer output changes.
        """

        if self.inputs is None:
            raise RuntimeError(
                "Forward propagation must run before backward propagation."
            )

        input_transposed = transpose(self.inputs)

        weights_gradient = matmul(
            input_transposed,
            output_gradient,
        )

        biases_gradient = sum_rows(output_gradient)

        weights_transposed = transpose(self.weights)

        input_gradient = matmul(
            output_gradient,
            weights_transposed,
        )

        weight_adjustment = scalar_multiply(
            weights_gradient,
            learning_rate,
        )

        bias_adjustment = scalar_multiply(
            biases_gradient,
            learning_rate,
        )

        self.weights = subtract(
            self.weights,
            weight_adjustment,
        )

        self.biases = subtract(
            self.biases,
            bias_adjustment,
        )

        return input_gradient


# =========================================================
# ACTIVATION LAYERS
# =========================================================

class SigmoidLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = sigmoid(inputs)

        return self.output

    def backward(self, output_gradient):
        derivative = sigmoid_derivative_from_output(
            self.output
        )

        return multiply_elementwise(
            output_gradient,
            derivative,
        )


class ReLULayer:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs

        return relu(inputs)

    def backward(self, output_gradient):
        derivative = relu_derivative(self.inputs)

        return multiply_elementwise(
            output_gradient,
            derivative,
        )


# =========================================================
# NEURAL NETWORK
# =========================================================

class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.05,
    ):
        self.learning_rate = learning_rate

        self.hidden_layer = DenseLayer(
            input_size,
            hidden_size,
        )

        self.hidden_activation = SigmoidLayer()

        self.output_layer = DenseLayer(
            hidden_size,
            output_size,
        )

    def forward(self, inputs):
        hidden_values = self.hidden_layer.forward(inputs)

        hidden_output = self.hidden_activation.forward(
            hidden_values
        )

        predictions = self.output_layer.forward(
            hidden_output
        )

        return predictions

    def backward(self, predictions, targets):
        loss_gradient = mean_squared_error_derivative(
            predictions,
            targets,
        )

        hidden_output_gradient = self.output_layer.backward(
            loss_gradient,
            self.learning_rate,
        )

        hidden_values_gradient = self.hidden_activation.backward(
            hidden_output_gradient
        )

        self.hidden_layer.backward(
            hidden_values_gradient,
            self.learning_rate,
        )

    def train(
        self,
        inputs,
        targets,
        epochs=10_000,
        print_every=500,
    ):
        for epoch in range(1, epochs + 1):
            predictions = self.forward(inputs)

            loss = mean_squared_error(
                predictions,
                targets,
            )

            self.backward(
                predictions,
                targets,
            )

            if epoch == 1 or epoch % print_every == 0:
                print(
                    f"Epoch {epoch:6d} | "
                    f"Loss: {loss:.10f}"
                )

    def predict(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        network_data = {
            "learning_rate": self.learning_rate,
            "hidden_weights": self.hidden_layer.weights,
            "hidden_biases": self.hidden_layer.biases,
            "output_weights": self.output_layer.weights,
            "output_biases": self.output_layer.biases,
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(network_data, file, indent=4)

    def load(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            network_data = json.load(file)

        self.learning_rate = network_data["learning_rate"]

        self.hidden_layer.weights = network_data[
            "hidden_weights"
        ]

        self.hidden_layer.biases = network_data[
            "hidden_biases"
        ]

        self.output_layer.weights = network_data[
            "output_weights"
        ]

        self.output_layer.biases = network_data[
            "output_biases"
        ]


# =========================================================
# ADDITION TRAINING DATA
# =========================================================

MAXIMUM_INPUT_NUMBER = 100.0
MAXIMUM_OUTPUT_NUMBER = MAXIMUM_INPUT_NUMBER * 2.0


def normalize_input(number):
    return number / MAXIMUM_INPUT_NUMBER


def normalize_output(number):
    return number / MAXIMUM_OUTPUT_NUMBER


def denormalize_output(number):
    return number * MAXIMUM_OUTPUT_NUMBER


def create_addition_training_data(number_of_examples):
    inputs = []
    targets = []

    for _ in range(number_of_examples):
        first_number = random.uniform(
            0,
            MAXIMUM_INPUT_NUMBER,
        )

        second_number = random.uniform(
            0,
            MAXIMUM_INPUT_NUMBER,
        )

        answer = first_number + second_number

        inputs.append(
            [
                normalize_input(first_number),
                normalize_input(second_number),
            ]
        )

        targets.append(
            [
                normalize_output(answer)
            ]
        )

    return inputs, targets


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():
    random.seed(42)

    training_inputs, training_targets = (
        create_addition_training_data(1_000)
    )

    network = NeuralNetwork(
        input_size=2,
        hidden_size=12,
        output_size=1,
        learning_rate=0.1,
    )

    print("Training the neural network...\n")

    network.train(
        training_inputs,
        training_targets,
        epochs=10_000,
        print_every=500,
    )

    network.save("addition_network.json")

    print("\nTraining completed.")
    print("The trained weights were saved.")
    print("Type 'exit' to stop.")

    while True:
        first_text = input("\nFirst number: ").strip()

        if first_text.lower() == "exit":
            break

        second_text = input("Second number: ").strip()

        if second_text.lower() == "exit":
            break

        try:
            first_number = float(first_text)
            second_number = float(second_text)
        except ValueError:
            print("Please enter valid numbers.")
            continue

        normalized_inputs = [
            [
                normalize_input(first_number),
                normalize_input(second_number),
            ]
        ]

        normalized_prediction = network.predict(
            normalized_inputs
        )[0][0]

        prediction = denormalize_output(
            normalized_prediction
        )

        print(
            f"Neural network prediction: {prediction:.4f}"
        )

        print(
            f"Correct answer: "
            f"{first_number + second_number:.4f}"
        )


if __name__ == "__main__":
    main()