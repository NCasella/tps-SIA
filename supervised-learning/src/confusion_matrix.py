

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
        header = "\t" + "\t".join(self.labels)
        rows = [f"{row}\t" + "\t".join(str(self.matrix[row][col]) for col in self.labels)
                for row in self.labels]
        return header + "\n" + "\n".join(rows)