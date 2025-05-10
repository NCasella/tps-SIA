

class ConfusionMatrix:
    def __init__(self, labels):
        self.labels = labels
        self.matrix = {row: {col: 0 for col in labels} for row in labels}

    def increment(self, actual, predicted):
        if actual not in self.labels or predicted not in self.labels:
            return
        self.matrix[actual][predicted] += 1

    def get(self, actual, predicted):
        return self.matrix[actual][predicted]

    def __str__(self):
        header = "\t" + "\t".join(str(label) for label in self.labels)
        rows = [f"{row}\t" + "\t".join(str(self.matrix[row][col]) for col in self.labels)
                for row in self.labels]
        return header + "\n" + "\n".join(rows)

    def accuracy(self):
        total = 0
        correct = 0
        for actual in self.matrix:
            for predicted in self.matrix[actual]:
                count = self.matrix[actual][predicted]
                total += count
                if actual == predicted:
                    correct += count
        return correct / total if total else 0.0

    def precision(self, label):
        tp = self.matrix[label][label]
        fp = sum(self.matrix[actual][label] for actual in self.matrix if actual != label)
        denominator = tp + fp
        return tp / denominator if denominator else 0.0

    def recall(self, label):
        tp = self.matrix[label][label]
        fn = sum(self.matrix[label][predicted] for predicted in self.matrix[label] if predicted != label)
        denominator = tp + fn
        return tp / denominator if denominator else 0.0

    def f1_score(self, label):
        precision = self.precision(label)
        recall = self.recall(label)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    def true_positives(self, label):
        return self.recall(self.matrix, label)

    def false_positives(self, label):
        fp = sum(self.matrix[actual][label] for actual in self.matrix if actual != label)
        tn = sum(self.matrix[actual][actual] for actual in self.matrix if actual != label)
        denominator = fp + tn
        return fp / denominator if denominator else 0.0