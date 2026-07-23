import math
import random
from .math_ops import (
    random_matrix,
    zeros,
    matmul,
    add_bias,
    transpose,
    sum_rows,
    matrix_shape,
    apply_function,
    multiply_elementwise,
)


def sigmoid_value(value):
    value = max(min(value, 500), -500)
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid(matrix):
    return apply_function(matrix, sigmoid_value)


def sigmoid_derivative_from_output(output):
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


def tanh(matrix):
    return apply_function(matrix, math.tanh)


def tanh_derivative_from_output(output):
    rows, cols = matrix_shape(output)
    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = 1.0 - output[i][j] ** 2
    return result


class DenseLayer:
    """A fully connected layer with SGD, momentum, and Adam updates."""

    def __init__(self, input_size, output_size):
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive.")

        # He-style initialization works well for ReLU-family activations.
        scale = math.sqrt(2.0 / input_size)
        self.weights = random_matrix(input_size, output_size, -scale, scale)
        self.biases = zeros(1, output_size)

        self.inputs = None
        self.output = None
        self.weights_gradient = None
        self.biases_gradient = None

        # Optimizer state. It is initialized lazily and reset after load().
        self._velocity_weights = zeros(input_size, output_size)
        self._velocity_biases = zeros(1, output_size)
        self._adam_m_weights = zeros(input_size, output_size)
        self._adam_m_biases = zeros(1, output_size)
        self._adam_v_weights = zeros(input_size, output_size)
        self._adam_v_biases = zeros(1, output_size)

    def forward(self, inputs):
        self.inputs = inputs
        weighted_values = matmul(inputs, self.weights)
        self.output = add_bias(weighted_values, self.biases)
        return self.output

    @staticmethod
    def _clip(value, limit):
        if limit is None:
            return value
        return max(-limit, min(limit, value))

    def _update_matrix(self, parameter, gradient, state_m, state_v,
                       learning_rate, optimizer, optimizer_params, step,
                       apply_weight_decay=False):
        rows, cols = matrix_shape(parameter)
        weight_decay = optimizer_params.get("weight_decay", 0.0) if apply_weight_decay else 0.0
        clip_value = optimizer_params.get("gradient_clip")

        if clip_value is not None and clip_value <= 0:
            raise ValueError("gradient_clip must be positive.")

        if optimizer == "sgd":
            for i in range(rows):
                for j in range(cols):
                    grad = gradient[i][j] + weight_decay * parameter[i][j]
                    grad = self._clip(grad, clip_value)
                    parameter[i][j] -= learning_rate * grad

        elif optimizer == "momentum":
            momentum = optimizer_params.get("momentum", 0.9)
            if not 0.0 <= momentum < 1.0:
                raise ValueError("momentum must be in [0, 1).")
            for i in range(rows):
                for j in range(cols):
                    grad = gradient[i][j] + weight_decay * parameter[i][j]
                    grad = self._clip(grad, clip_value)
                    state_m[i][j] = momentum * state_m[i][j] - learning_rate * grad
                    parameter[i][j] += state_m[i][j]

        elif optimizer == "adam":
            beta1 = optimizer_params.get("beta1", 0.9)
            beta2 = optimizer_params.get("beta2", 0.999)
            epsilon = optimizer_params.get("epsilon", 1e-8)
            if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
                raise ValueError("Adam beta1 and beta2 must be in [0, 1).")
            if epsilon <= 0:
                raise ValueError("Adam epsilon must be positive.")

            correction1 = 1.0 - beta1 ** step
            correction2 = 1.0 - beta2 ** step

            for i in range(rows):
                for j in range(cols):
                    grad = gradient[i][j] + weight_decay * parameter[i][j]
                    grad = self._clip(grad, clip_value)
                    state_m[i][j] = beta1 * state_m[i][j] + (1.0 - beta1) * grad
                    state_v[i][j] = beta2 * state_v[i][j] + (1.0 - beta2) * (grad ** 2)
                    m_hat = state_m[i][j] / correction1
                    v_hat = state_v[i][j] / correction2
                    parameter[i][j] -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
        else:
            raise ValueError("optimizer must be 'sgd', 'momentum', or 'adam'.")

    def backward(self, output_gradient, learning_rate, optimizer="sgd",
                 optimizer_params=None, step=1, update=True):
        if self.inputs is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")

        optimizer_params = optimizer_params or {}
        self.weights_gradient = matmul(transpose(self.inputs), output_gradient)
        self.biases_gradient = sum_rows(output_gradient)

        # Compute this before changing the weights.
        input_gradient = matmul(output_gradient, transpose(self.weights))

        if update:
            self._update_matrix(
                self.weights,
                self.weights_gradient,
                self._velocity_weights if optimizer == "momentum" else self._adam_m_weights,
                self._adam_v_weights,
                learning_rate,
                optimizer,
                optimizer_params,
                step,
                apply_weight_decay=True,
            )
            self._update_matrix(
                self.biases,
                self.biases_gradient,
                self._velocity_biases if optimizer == "momentum" else self._adam_m_biases,
                self._adam_v_biases,
                learning_rate,
                optimizer,
                optimizer_params,
                step,
                apply_weight_decay=False,
            )

        return input_gradient

    def reset_optimizer_state(self):
        input_size = len(self.weights)
        output_size = len(self.weights[0])
        self._velocity_weights = zeros(input_size, output_size)
        self._velocity_biases = zeros(1, output_size)
        self._adam_m_weights = zeros(input_size, output_size)
        self._adam_m_biases = zeros(1, output_size)
        self._adam_v_weights = zeros(input_size, output_size)
        self._adam_v_biases = zeros(1, output_size)

    def parameter_count(self):
        return len(self.weights) * len(self.weights[0]) + len(self.biases[0])

    def get_config(self):
        return {
            "type": "DenseLayer",
            "input_size": len(self.weights),
            "output_size": len(self.weights[0])
        }


class SigmoidLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = sigmoid(inputs)
        return self.output

    def backward(self, output_gradient):
        if self.output is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")
        derivative = sigmoid_derivative_from_output(self.output)
        return multiply_elementwise(output_gradient, derivative)

    def get_config(self):
        return {"type": "SigmoidLayer"}


class ReLULayer:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return relu(inputs)

    def backward(self, output_gradient):
        if self.inputs is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")
        derivative = relu_derivative(self.inputs)
        return multiply_elementwise(output_gradient, derivative)

    def get_config(self):
        return {"type": "ReLULayer"}


class LeakyReLULayer:
    def __init__(self, alpha=0.01):
        if alpha < 0:
            raise ValueError("alpha cannot be negative.")
        self.alpha = alpha
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return apply_function(inputs, lambda value: value if value > 0 else self.alpha * value)

    def backward(self, output_gradient):
        if self.inputs is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")
        derivative = apply_function(
            self.inputs,
            lambda value: 1.0 if value > 0 else self.alpha,
        )
        return multiply_elementwise(output_gradient, derivative)

    def get_config(self):
        return {"type": "LeakyReLULayer", "alpha": self.alpha}


class TanhLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = tanh(inputs)
        return self.output

    def backward(self, output_gradient):
        if self.output is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")
        return multiply_elementwise(output_gradient, tanh_derivative_from_output(self.output))

    def get_config(self):
        return {"type": "TanhLayer"}


class SoftmaxLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        rows, cols = matrix_shape(inputs)
        self.output = zeros(rows, cols)

        for i in range(rows):
            max_val = max(inputs[i])
            exp_sum = 0.0

            for j in range(cols):
                value = math.exp(inputs[i][j] - max_val)
                self.output[i][j] = value
                exp_sum += value

            for j in range(cols):
                self.output[i][j] /= exp_sum

        return self.output

    def backward(self, output_gradient):
        if self.output is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")
        rows, cols = matrix_shape(self.output)
        input_gradient = zeros(rows, cols)

        # Jacobian-vector product for each sample, without building a full Jacobian.
        for i in range(rows):
            dot_product = 0.0
            for j in range(cols):
                dot_product += self.output[i][j] * output_gradient[i][j]

            for k in range(cols):
                input_gradient[i][k] = self.output[i][k] * (
                    output_gradient[i][k] - dot_product
                )

        return input_gradient

    def get_config(self):
        return {"type": "SoftmaxLayer"}


class DropoutLayer:
    def __init__(self, rate=0.5):
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1).")
        self.rate = rate
        self.mask = None
        self.is_training = False

    def forward(self, inputs):
        if not self.is_training:
            return inputs
            
        rows, cols = matrix_shape(inputs)
        self.mask = zeros(rows, cols)
        output = zeros(rows, cols)
        
        scale = 1.0 / (1.0 - self.rate)
        for i in range(rows):
            for j in range(cols):
                if random.random() > self.rate:
                    self.mask[i][j] = scale
                    output[i][j] = inputs[i][j] * scale
        return output

    def backward(self, output_gradient):
        if not self.is_training:
            return output_gradient
            
        rows, cols = matrix_shape(output_gradient)
        input_gradient = zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                input_gradient[i][j] = output_gradient[i][j] * self.mask[i][j]
        return input_gradient

    def get_config(self):
        return {"type": "DropoutLayer", "rate": self.rate}
