import math
from .math_ops import (
    random_matrix, zeros, matmul, add_bias, transpose, 
    sum_rows, scalar_multiply, subtract, matrix_shape, 
    apply_function, multiply_elementwise
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

class DenseLayer:
    def __init__(self, input_size, output_size):
        scale = math.sqrt(2.0 / input_size)
        self.weights = random_matrix(input_size, output_size, -scale, scale)
        self.biases = zeros(1, output_size)
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        weighted_values = matmul(inputs, self.weights)
        self.output = add_bias(weighted_values, self.biases)
        return self.output

    def backward(self, output_gradient, learning_rate):
        if self.inputs is None:
            raise RuntimeError("Forward propagation must run before backward propagation.")
        input_transposed = transpose(self.inputs)
        weights_gradient = matmul(input_transposed, output_gradient)
        biases_gradient = sum_rows(output_gradient)
        weights_transposed = transpose(self.weights)
        input_gradient = matmul(output_gradient, weights_transposed)
        weight_adjustment = scalar_multiply(weights_gradient, learning_rate)
        bias_adjustment = scalar_multiply(biases_gradient, learning_rate)
        self.weights = subtract(self.weights, weight_adjustment)
        self.biases = subtract(self.biases, bias_adjustment)
        return input_gradient

class SigmoidLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = sigmoid(inputs)
        return self.output

    def backward(self, output_gradient):
        derivative = sigmoid_derivative_from_output(self.output)
        return multiply_elementwise(output_gradient, derivative)

class ReLULayer:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return relu(inputs)

    def backward(self, output_gradient):
        derivative = relu_derivative(self.inputs)
        return multiply_elementwise(output_gradient, derivative)

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
                val = math.exp(inputs[i][j] - max_val)
                self.output[i][j] = val
                exp_sum += val
                
            for j in range(cols):
                self.output[i][j] /= exp_sum
                
        return self.output

    def backward(self, output_gradient):
        rows, cols = matrix_shape(self.output)
        input_gradient = zeros(rows, cols)
        
        for i in range(rows):
            dot_product = 0.0
            for j in range(cols):
                dot_product += self.output[i][j] * output_gradient[i][j]
                
            for k in range(cols):
                input_gradient[i][k] = self.output[i][k] * output_gradient[i][k] - self.output[i][k] * dot_product
                
        return input_gradient
