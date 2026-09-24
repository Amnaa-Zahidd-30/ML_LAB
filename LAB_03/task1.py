"""
Activity 1 — Implement Entropy
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


if __name__ == "__main__":
    labels1 = ["Pass", "Pass", "Pass", "Pass"]
    labels2 = ["Pass", "Pass", "Fail", "Fail"]
    labels3 = ["Pass", "Fail", "Pass", "Fail"]

    print(f"labels1 (all Pass)      -> entropy = {entropy(labels1)}")
    print(f"labels2 (2 Pass/2 Fail) -> entropy = {entropy(labels2)}")
    print(f"labels3 (2 Pass/2 Fail) -> entropy = {entropy(labels3)}")

    print(
        "\nExplanation: labels1 has only one class present, so there is "
        "zero uncertainty and entropy = 0. labels2 and labels3 are both "
        "split 50/50 between Pass and Fail, so entropy = 1 bit in both "
        "cases — entropy depends only on class proportions, not on the "
        "order the labels appear in."
    )