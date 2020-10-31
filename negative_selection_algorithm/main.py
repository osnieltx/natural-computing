import random
from typing import List

# Getting data
with open('animals.txt', 'r') as file:
    rows = [[int(x) for x in line.split()] for line in file]
    train_data: List[List] = [list(col) for col in zip(*rows)]

test_data = [
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]


def learn(detectors_set_size: int, self_examples: List[List[int]], affinity_trashhold: int):
    detectors_set = set()
    antigen_size = len(self_examples[0])
    possible_detectors = [i for i in range(2**antigen_size)]
    while len(detectors_set) < detectors_set_size:
        new_detector_index = random.randint(0, len(possible_detectors)-1)
        new_detector = tuple(int(c) for c in f"{possible_detectors.pop(new_detector_index):b}")
        for self_ex in self_examples:
            similarity = []
            pivot = 0
            for index, value in enumerate(new_detector):
                if value != self_ex[index]:
                    if pivot != index:
                        similarity.append(new_detector[pivot:index])
                    pivot = index + 1
            similarity = max(len(s) for s in similarity) if similarity else 0
            if similarity >= affinity_trashhold:
                break
        else:
            detectors_set.add(new_detector)
    return detectors_set


def nonself_count(detectors_set: List[List], some_set: List[List], affinity_trashhold: int):
    nonself_qtt = 0
    for some in some_set:
        for detector in detectors_set:
            similarity = []
            pivot = 0
            for index, value in enumerate(detector):
                if value != some[index]:
                    if pivot != index:
                        similarity.append(detector[pivot:index])
                    pivot = index + 1
            similarity = max(len(s) for s in similarity) if similarity else 0
            if similarity >= affinity_trashhold:
                nonself_qtt += 1
                break
    return nonself_qtt


# Learn
for affinity_trashold in range(len(test_data[0]), 0, -1):
    print(f"Testing affinity trashold: {affinity_trashold}... ", end='')
    detectors = learn(2000, train_data, affinity_trashold)

    # Classify
    nonself_qtt = nonself_count(detectors, test_data, affinity_trashold)
    detection_rate = nonself_qtt/len(test_data)
    print(f"detection rate: {detection_rate}", end='')
    print(f", FP ratio: {1-detection_rate}")
