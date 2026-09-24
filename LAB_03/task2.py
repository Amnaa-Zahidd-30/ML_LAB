"""
Activity 2 — Implement Mutual Information
"""

import math


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

    labels = [row["Label"] for row in dataset]
    print(f"H(Y) for the full dataset = {entropy(labels):.4f}\n")

    for feature in features:
        mi = mutual_information(dataset, feature)
        print(feature, mi)

    best_feature = max(features, key=lambda f: mutual_information(dataset, f))
    print("\nBest feature:", best_feature)