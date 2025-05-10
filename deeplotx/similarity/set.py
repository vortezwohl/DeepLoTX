from deeplotx.similarity import bias


def jaccard_similarity(set1: set, set2: set) -> float:
    return (len(set1.intersection(set2)) + bias) / (len(set1.union(set2)) + bias)


def overlap_coefficient(set1: set, set2: set) -> float:
    return (len(set1.intersection(set2)) + bias) / (min(len(set1), len(set2)) + bias)


def dice_coefficient(set1: set, set2: set) -> float:
    return (2 * len(set1.intersection(set2)) + bias) / (len(set1) + len(set2) + bias)


def ochiai_similarity(set1: set, set2: set) -> float:
    intersection = len(set1.intersection(set2))
    product = len(set1) ** 0.5 * len(set2) ** 0.5
    return (intersection + bias) / (product + bias)
