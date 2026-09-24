"""
Activity 4 — Implement Recursive Training
"""

import math
from collections import Counter


class Node:
    def __init__(self, node_type, label=None, feature=None):
        self.type = node_type
        self.label = label
        self.feature = feature
        self.branches = {}


def entropy(labels):
    if len(labels) == 0:
        return 0

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    total = len(labels)
    result = 0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)

    return result


def conditional_entropy(dataset, feature, label_col="Label"):
    total = len(dataset)
    weighted_entropy = 0
    values = set(row[feature] for row in dataset)

    for value in values:
        subset = [
            row[label_col]
            for row in dataset
            if row[feature] == value
        ]
        weight = len(subset) / total
        weighted_entropy += weight * entropy(subset)

    return weighted_entropy


def mutual_information(dataset, feature, label_col="Label"):
    labels = [row[label_col] for row in dataset]
    h_y = entropy(labels)
    h_y_given_x = conditional_entropy(dataset, feature, label_col)
    return h_y - h_y_given_x


def majority_label(dataset, label_col="Label"):
    labels = [row[label_col] for row in dataset]
    return Counter(labels).most_common(1)[0][0]


def train(dataset, features, label_col="Label"):
    # Base Case 1: all labels are identical
    labels = [row[label_col] for row in dataset]
    if len(set(labels)) == 1:
        return Node("leaf", label=labels[0])

    # Base Case 2: no features remain
    if len(features) == 0:
        return Node("leaf", label=majority_label(dataset, label_col))

    # Select feature with maximum Mutual Information
    best_feature = max(
        features,
        key=lambda f: mutual_information(dataset, f, label_col)
    )

    node = Node("internal", feature=best_feature)

    values = set(row[best_feature] for row in dataset)
    remaining_features = [f for f in features if f != best_feature]

    for value in values:
        subset = [row for row in dataset if row[best_feature] == value]
        child = train(subset, remaining_features, label_col)
        node.branches[value] = child

    return node


def print_tree(node, depth=0):
    indent = "  " * depth
    if node.type == "leaf":
        print(f"{indent}Leaf -> {node.label}")
    else:
        print(f"{indent}[{node.feature}]")
        for value, child in node.branches.items():
            print(f"{indent}  {node.feature} = {value}:")
            print_tree(child, depth + 2)


dataset = [
    {"Attendance": "High", "Study": "High", "Assignment": "Yes", "Label": "Pass"},
    {"Attendance": "High", "Study": "Low",  "Assignment": "Yes", "Label": "Pass"},
    {"Attendance": "High", "Study": "Low",  "Assignment": "No",  "Label": "Pass"},
    {"Attendance": "Low",  "Study": "High", "Assignment": "Yes", "Label": "Pass"},
    {"Attendance": "Low",  "Study": "Low",  "Assignment": "No",  "Label": "Fail"},
    {"Attendance": "Low",  "Study": "Low",  "Assignment": "Yes", "Label": "Fail"},
    {"Attendance": "High", "Study": "High", "Assignment": "No",  "Label": "Pass"},
    {"Attendance": "Low",  "Study": "High", "Assignment": "No",  "Label": "Fail"},
]


if __name__ == "__main__":
    features = ["Attendance", "Study", "Assignment"]
    tree = train(dataset, features)

    print("Trained tree structure:")
    print_tree(tree)

    new_student = {"Attendance": "High", "Study": "Low", "Assignment": "No"}

    def predict(tree, x):
        current = tree
        while current.type != "leaf":
            value = x[current.feature]
            current = current.branches[value]
        return current.label

    prediction = predict(tree, new_student)
    print("\nPrediction for", new_student, "->", prediction)