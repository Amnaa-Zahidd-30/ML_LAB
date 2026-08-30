\
from collections import Counter, defaultdict
import random

def classification_error(y_true, y_pred):
    mistakes = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual != predicted:
            mistakes += 1
    return mistakes / len(y_true)

# TASK 1 - Prepare the Dataset

X_train = [
    ["Y", "N", "N", "N"],
    ["N", "Y", "N", "N"],
    ["Y", "Y", "N", "N"],
    ["Y", "N", "Y", "Y"],
    ["N", "Y", "Y", "N"]
]
y_train = ["-", "-", "+", "-", "+"]

X_test = [
    ["Y", "Y", "Y", "N"],
    ["N", "N", "N", "N"],
    ["Y", "N", "N", "Y"]
]
y_test = ["+", "-", "-"]

# TASK 2 - Implement Majority Vote

class MajorityVoteClassifier:
    def __init__(self):
        self.majority_label = None

    def fit(self, X, y):
        counts = Counter(y)
        self.majority_label = counts.most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.majority_label for _ in X]


print("=" * 60)
print("TASK 2: Majority Vote Classifier")
print("=" * 60)

majority = MajorityVoteClassifier()
majority.fit(X_train, y_train)

majority_train_pred = majority.predict(X_train)
majority_test_pred = majority.predict(X_test)

majority_train_error = classification_error(y_train, majority_train_pred)
majority_test_error = classification_error(y_test, majority_test_pred)

print("Majority label learned:", majority.majority_label)
print("Train predictions:", majority_train_pred)
print("Test predictions: ", majority_test_pred)
print("Training error:", majority_train_error)
print("Test error:    ", majority_test_error)
print()

# TASK 3 - Implement Memorizer

class MemorizerClassifier:
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


print("=" * 60)
print("TASK 3: Memorizer Classifier")
print("=" * 60)

memorizer = MemorizerClassifier()
memorizer.fit(X_train, y_train)

memorizer_train_pred = memorizer.predict(X_train)
memorizer_train_error = classification_error(y_train, memorizer_train_pred)

print("Train predictions:", memorizer_train_pred)
print("Training error (should be 0.0):", memorizer_train_error)

memorizer_test_pred = memorizer.predict(X_test)
memorizer_test_error = classification_error(y_test, memorizer_test_pred)

print("Test predictions (unseen combos -> random fallback):", memorizer_test_pred)
print("Test error:", memorizer_test_error)
print("Observation: training error is 0 because every X_train row is memorized,")
print("but test error is unreliable/random because none of X_test's feature")
print("combinations exist in memory, so it falls back to random.choice().")
print()

# TASK 4 - Implement Decision Stump

class DecisionStumpClassifier:
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


print("=" * 60)
print("TASK 4: Decision Stump Classifier (compare two feature indices)")
print("=" * 60)

for idx in [0, 2]:  # feature 0 = "hives", feature 2 = "red eye"
    stump = DecisionStumpClassifier(feature_index=idx)
    stump.fit(X_train, y_train)

    train_pred = stump.predict(X_train)
    test_pred = stump.predict(X_test)

    train_err = classification_error(y_train, train_pred)
    test_err = classification_error(y_test, test_pred)

    print(f"Feature index {idx}")
    print("  Learned rules:", stump.rules)
    print("  Default label:", stump.default_label)
    print("  Train predictions:", train_pred, "| Train error:", train_err)
    print("  Test predictions: ", test_pred, "| Test error:", test_err)
    print()


# TASK 5 - Compare the Algorithms

print("=" * 60)
print("TASK 5: Compare Majority Vote, Memorizer, Decision Stump")
print("=" * 60)

majority = MajorityVoteClassifier()
majority.fit(X_train, y_train)

memorizer = MemorizerClassifier()
memorizer.fit(X_train, y_train)

stump = DecisionStumpClassifier(feature_index=2)
stump.fit(X_train, y_train)

models = {
    "Majority Vote": majority,
    "Memorizer": memorizer,
    "Decision Stump": stump
}

for name, model in models.items():
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_error = classification_error(y_train, train_pred)
    test_error = classification_error(y_test, test_pred)

    print(name)
    print("Training error:", train_error)
    print("Test error:", test_error)
    print()


