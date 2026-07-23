import random

def matrix_shape(matrix):
    return len(matrix), len(matrix[0])

def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def random_matrix(rows, cols, minimum=-1.0, maximum=1.0):
    return [[random.uniform(minimum, maximum) for _ in range(cols)] for _ in range(rows)]

def matmul(A, B):
    if not A or not B:
        raise ValueError("Matrices cannot be empty.")
    if len(A[0]) != len(B):
        raise ValueError(f"Cannot multiply matrices with shapes {len(A)}x{len(A[0])} and {len(B)}x{len(B[0])}.")
    rows, cols, inner = len(A), len(B[0]), len(B)
    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    return result

def transpose(matrix):
    rows, cols = len(matrix), len(matrix[0])
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
    rows, cols = matrix_shape(matrix)
    if len(bias) != 1 or len(bias[0]) != cols:
        raise ValueError("Bias shape does not match matrix.")
    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[i][j] + bias[0][j]
    return result

def sum_rows(matrix):
    rows, cols = matrix_shape(matrix)
    result = zeros(1, cols)
    for i in range(rows):
        for j in range(cols):
            result[0][j] += matrix[i][j]
    return result
