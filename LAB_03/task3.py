"""
Activity 3 — Implement Prediction
"""


class Node:
    def __init__(self, node_type, label=None, feature=None):
        self.type = node_type       # "leaf" or "internal"
        self.label = label          # used by leaf nodes
        self.feature = feature      # used by internal nodes
        self.branches = {}          # feature value -> child Node


def predict(tree, x):
    current = tree
    while current.type != "leaf":
        feature = current.feature
        value = x[feature]
        current = current.branches[value]
    return current.label


if __name__ == "__main__":
    # Build the small tree manually (same example as the lab handout)
    fail_leaf = Node("leaf", label="Fail")
    pass_leaf = Node("leaf", label="Pass")

    study_node = Node("internal", feature="Study")
    study_node.branches["Low"] = fail_leaf
    study_node.branches["High"] = pass_leaf

    root = Node("internal", feature="Attendance")
    root.branches["Low"] = fail_leaf
    root.branches["High"] = study_node

    # Test at least three new students
    test_students = [
        {"Attendance": "High", "Study": "High"},
        {"Attendance": "High", "Study": "Low"},
        {"Attendance": "Low",  "Study": "High"},
    ]

    for i, student in enumerate(test_students, start=1):
        result = predict(root, student)
        print(f"Student {i}: {student} -> Prediction: {result}")