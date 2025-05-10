import math

def _accuarcy(matrix):
    numerator = 0
    for i, line in enumerate(matrix):
        for j, column in enumerate(line):
            numerator += matrix[i][j]
    return numerator / sum(matrix)

def _precision(matrix, index):
    denominator = 0
    for line in matrix:
        denominator += line[index]
    return matrix[index][index] / denominator

def _recall(matrix, index):
    denominator = 0
    for i in matrix.len:
        denominator += matrix[i][index]
    return matrix[index][index] / denominator

def _f1_score(matrix, index):
    precision = _precision(matrix, index)
    recall = _recall(matrix, index)
    return (2 * precision * recall) / (precision + recall)

def _true_positives(matrix, index):
    return _recall(matrix, index)

def _false_positives(matrix, index):
    denominator = 0
    for i in matrix.len:
        denominator += matrix[i][index]
    numerator = denominator - matrix[index][index]
    return numerator / denominator