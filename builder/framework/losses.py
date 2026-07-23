import math
from .math_ops import matrix_shape, zeros

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
            result[i][j] = (2.0 * (predictions[i][j] - targets[i][j]) / number_of_values)
    return result

def binary_cross_entropy(predictions, targets):
    rows, cols = matrix_shape(predictions)
    total_error = 0.0
    number_of_values = rows * cols
    
    for i in range(rows):
        for j in range(cols):
            p = max(min(predictions[i][j], 1.0 - 1e-7), 1e-7)
            y = targets[i][j]
            total_error += - (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
            
    return total_error / number_of_values

def binary_cross_entropy_derivative(predictions, targets):
    rows, cols = matrix_shape(predictions)
    number_of_values = rows * cols
    result = zeros(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            p = max(min(predictions[i][j], 1.0 - 1e-7), 1e-7)
            y = targets[i][j]
            result[i][j] = ((p - y) / (p * (1.0 - p))) / number_of_values
            
    return result

def categorical_cross_entropy(predictions, targets):
    rows, cols = matrix_shape(predictions)
    total_error = 0.0
    
    for i in range(rows):
        for j in range(cols):
            p = max(min(predictions[i][j], 1.0 - 1e-7), 1e-7)
            total_error += - (targets[i][j] * math.log(p))
            
    return total_error / rows

def categorical_cross_entropy_derivative(predictions, targets):
    rows, cols = matrix_shape(predictions)
    result = zeros(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            p = max(min(predictions[i][j], 1.0 - 1e-7), 1e-7)
            result[i][j] = - (targets[i][j] / p) / rows
            
    return result
