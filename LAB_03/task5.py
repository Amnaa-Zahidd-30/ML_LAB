"""
Activity 5 — Test the Trained Tree
"""

import math
from collections import Counter


class Node:
    def __init__(self, node_type, label=None, feature=None):
        self.type = node_type
        self.label = label
        self.feature = feature
        self.branches = {}


def predict(tree, x):
    current = tree
    while current.type != "leaf":
        feature = current.feature
        value = x[feature]
        current = current.branches[value]
    return current.label


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
    labels = [row[label_col] for row in dataset]
    if len(set(labels)) == 1:
        return Node("leaf", label=labels[0])

    if len(features) == 0:
        return Node("leaf", label=majority_label(dataset, label_col))

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

    # 5 new student records
    new_students = [
        {"Attendance": "High", "Study": "Low",  "Assignment": "No"},
        {"Attendance": "Low",  "Study": "Low",  "Assignment": "No"},
        {"Attendance": "High", "Study": "High", "Assignment": "Yes"},
        {"Attendance": "Low",  "Study": "High", "Assignment": "Yes"},
        {"Attendance": "Low",  "Study": "Low",  "Assignment": "Yes"},
    ]

    for i, student in enumerate(new_students, start=1):
        result = predict(tree, student)
        print(f"New student {i}: {student} -> Prediction: {result}")