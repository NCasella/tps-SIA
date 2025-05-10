def _accuracy(matrix):
    total = 0
    correct = 0
    for actual in matrix:
        for predicted in matrix[actual]:
            count = matrix[actual][predicted]
            total += count
            if actual == predicted:
                correct += count
    return correct / total if total else 0.0


def _precision(matrix, label):
    tp = matrix[label][label]
    fp = sum(matrix[actual][label] for actual in matrix if actual != label)
    denominator = tp + fp
    return tp / denominator if denominator else 0.0


def _recall(matrix, label):
    tp = matrix[label][label]
    fn = sum(matrix[label][predicted] for predicted in matrix[label] if predicted != label)
    denominator = tp + fn
    return tp / denominator if denominator else 0.0


def _f1_score(matrix, label):
    precision = _precision(matrix, label)
    recall = _recall(matrix, label)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def _true_positives(matrix, label):
    return _recall(matrix, label)


def _false_positives(matrix, label):
    fp = sum(matrix[actual][label] for actual in matrix if actual != label)
    tn = sum(matrix[actual][actual] for actual in matrix if actual != label)
    denominator = fp + tn
    return fp / denominator if denominator else 0.0