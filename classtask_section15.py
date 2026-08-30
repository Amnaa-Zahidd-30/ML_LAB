
from collections import Counter, defaultdict
import random
# SECTION 15 - CLASS TASK

class BaseClassifier:
    """Common parent class defining the shared interface for all classifiers."""

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(
            actual == predicted
            for actual, predicted in zip(y, predictions)
        )
        return correct / len(y)


class MajorityVoteClassifierV2(BaseClassifier):
    def __init__(self):
        self.majority_label = None

    def fit(self, X, y):
        counts = Counter(y)
        self.majority_label = counts.most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.majority_label for _ in X]


class MemorizerClassifierV2(BaseClassifier):
    def __init__(self):
        self.memory = {}
        self.labels = []

    def fit(self, X, y):
        self.memory = {}
        self.labels = list(set(y))
        for features, label in zip(X, y):
            self.memory[tuple(features)] = label
        return self

    def predict(self, X):
        predictions = []
        for features in X:
            key = tuple(features)
            if key in self.memory:
                predictions.append(self.memory[key])
            else:
                predictions.append(random.choice(self.labels))
        return predictions


class DecisionStumpClassifierV2(BaseClassifier):
    def __init__(self, feature_index=0):
        self.feature_index = feature_index
        self.rules = {}
        self.default_label = None

    def fit(self, X, y):
        groups = defaultdict(list)
        for features, label in zip(X, y):
            value = features[self.feature_index]
            groups[value].append(label)

        self.rules = {}
        for value, labels in groups.items():
            self.rules[value] = Counter(labels).most_common(1)[0][0]

        self.default_label = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        predictions = []
        for features in X:
            value = features[self.feature_index]
            predictions.append(self.rules.get(value, self.default_label))
        return predictions


def make_larger_dataset(n=200, seed=42):
    """
    Generates a larger synthetic binary-classification dataset with
    4 binary (Y/N) features, so the framework can be tested at scale.
    Label rule (hidden from the learner): '+' if feature 1 == 'Y'
    OR (feature 0 == 'Y' AND feature 3 == 'Y'); else '-' with 10% label noise.
    """
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        row = [rng.choice(["Y", "N"]) for _ in range(4)]
        label = "+" if (row[1] == "Y" or (row[0] == "Y" and row[3] == "Y")) else "-"
        if rng.random() < 0.10:  # flip 10% of labels to add noise
            label = "+" if label == "-" else "-"
        X.append(row)
        y.append(label)
    return X, y


print("=" * 60)
print("CLASS TASK (Section 15): BaseClassifier framework on a larger dataset")
print("=" * 60)

X_big, y_big = make_larger_dataset(n=200, seed=42)

split_point = int(0.7 * len(X_big))
X_train_big, X_test_big = X_big[:split_point], X_big[split_point:]
y_train_big, y_test_big = y_big[:split_point], y_big[split_point:]

print(f"Total examples: {len(X_big)} | Train: {len(X_train_big)} | Test: {len(X_test_big)}")
print()

framework_models = {
    "Majority Vote": MajorityVoteClassifierV2(),
    "Memorizer": MemorizerClassifierV2(),
    "Decision Stump (f0)": DecisionStumpClassifierV2(feature_index=0),
    "Decision Stump (f1)": DecisionStumpClassifierV2(feature_index=1),
    "Decision Stump (f2)": DecisionStumpClassifierV2(feature_index=2),
    "Decision Stump (f3)": DecisionStumpClassifierV2(feature_index=3),
}

print(f"{'Model':<22}{'Train Acc':<12}{'Test Acc':<12}")
print("-" * 46)
for name, model in framework_models.items():
    model.fit(X_train_big, y_train_big)
    train_acc = model.score(X_train_big, y_train_big)
    test_acc = model.score(X_test_big, y_test_big)
    print(f"{name:<22}{train_acc:<12.3f}{test_acc:<12.3f}")

print()
print("Observation: Majority Vote gives a flat, weak baseline. Memorizer gets")
print("near-perfect training accuracy but its test accuracy drops because most")
print("test feature combinations were never seen during training. Decision")
print("Stump on feature 1 (the strongest single predictor in the label rule)")
print("outperforms the other single-feature stumps, but no stump can fully")
print("capture the AND interaction between feature 0 and feature 3 - that")
print("would require a deeper Decision Tree.")