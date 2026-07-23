import random


def validate_matrix(matrix, name="matrix"):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of rows.")
    if not isinstance(matrix[0], list) or not matrix[0]:
        raise ValueError(f"{name} must contain non-empty rows.")
    width = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{name} must be rectangular.")
    return len(matrix), width


def matrix_shape(matrix):
    return validate_matrix(matrix)


def zeros(rows, cols):
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows, cols, minimum=-1.0, maximum=1.0):
    if minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum.")
    return [[random.uniform(minimum, maximum) for _ in range(cols)] for _ in range(rows)]


def matmul(A, B):
    rows_a, cols_a = validate_matrix(A, "A")
    rows_b, cols_b = validate_matrix(B, "B")
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply shapes {rows_a}x{cols_a} and {rows_b}x{cols_b}.")
    result = zeros(rows_a, cols_b)
    bt = transpose(B)
    for i, row in enumerate(A):
        for j, col in enumerate(bt):
            result[i][j] = sum(x * y for x, y in zip(row, col))
    return result


def transpose(matrix):
    rows, cols = validate_matrix(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def _same_shape(A, B):
    shape_a = validate_matrix(A, "A")
    shape_b = validate_matrix(B, "B")
    if shape_a != shape_b:
        raise ValueError(f"Matrices must have the same shape, got {shape_a} and {shape_b}.")
    return shape_a


def add(A, B):
    rows, cols = _same_shape(A, B)
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]


def subtract(A, B):
    rows, cols = _same_shape(A, B)
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]


def multiply_elementwise(A, B):
    rows, cols = _same_shape(A, B)
    return [[A[i][j] * B[i][j] for j in range(cols)] for i in range(rows)]


def scalar_multiply(matrix, scalar):
    rows, cols = validate_matrix(matrix)
    return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]


def apply_function(matrix, function):
    rows, cols = validate_matrix(matrix)
    return [[function(matrix[i][j]) for j in range(cols)] for i in range(rows)]


def add_bias(matrix, bias):
    rows, cols = validate_matrix(matrix)
    bias_rows, bias_cols = validate_matrix(bias, "bias")
    if bias_rows != 1 or bias_cols != cols:
        raise ValueError(f"Bias must have shape 1x{cols}.")
    return [[matrix[i][j] + bias[0][j] for j in range(cols)] for i in range(rows)]


def sum_rows(matrix):
    rows, cols = validate_matrix(matrix)
    return [[sum(matrix[i][j] for i in range(rows)) for j in range(cols)]]
