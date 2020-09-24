import random
import struct
from codecs import decode
from copy import copy, deepcopy
from math import sin, pi


def problem(x: float):
    try:
        return pow(2, -2 * pow((x-0.1)/0.9, 2)) * pow(sin(5 * pi * x), 6)
    except OverflowError:
        return 0.


def fitness(x: str): return problem(bin_to_float(x))


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)


def reproduce(p: float):
    specimens = list(population.keys())
    offspring = {}
    for index, specimen_a in enumerate(specimens):
        for second_index in range(index+1, len(specimens)):
            should_exchange_genes = random.random() * 100 <= p
            if should_exchange_genes:
                cut_point = random.randint(0, len(specimens[0])-1)
                offspring[f'{specimen_a[:cut_point]}{specimens[second_index][cut_point:]}'] = 0.
                offspring[f'{specimens[second_index][:cut_point]}{specimen_a[cut_point:]}'] = 0.

    population.update(offspring)


def mutate(p: float):
    new_population = {}
    for specimen in deepcopy(population):
        original_specimen = copy(specimen)
        for i in range(len(specimen)):
            should_mutate = random.random() * 100 <= p
            if should_mutate:
                if i != len(specimen) - 1:
                    end = specimen[i+1:]
                else:
                    end = ''
                specimen = f"{specimen[:i]}{'0' if specimen[i] == '1' else '1'}{end}"
        if specimen != original_specimen:
            value = population.pop(original_specimen)
            new_population[specimen] = value
    population.update(new_population)

def evaluate():
    for p in population:
        population[p] = fitness(p)


def select():
    global population
    list_population = [(specimen, fit) for specimen, fit in population.items()]
    list_population.sort(key=lambda x: x[1], reverse=True)
    total_fit = sum(population.values())
    new_population = {}
    for _ in range(100):
        selected = random.random() * total_fit
        index, fit_count = 0, list_population[0][1]
        while fit_count < selected:
            index += 1
            fit_count += list_population[index][1]
        new_population[list_population[index][0]] = list_population[index][1]
    population = new_population


population = {}
for _ in range(100):
    specimen = (random.random() * 4) - 2
    population[float_to_bin(specimen)] = problem(specimen)
    print(len(list(population.keys())[-1]))

t = 0
best_specimen_evolution = []
mean_fitness_evolution = []
generations = 100
while t < generations:
    reproduce(50)
    mutate(0.1)
    evaluate()
    select()
    best_specimen_evolution.append(max(population.values()))
    mean_fitness_evolution.append(sum(population.values())/len(population.values()))
    t += 1

import matplotlib.pyplot as plt
plt.plot(range(generations), best_specimen_evolution, 'r', label='Melhor fitness da geração')
plt.plot(range(generations), mean_fitness_evolution, 'b', label='Fitness médio da população')
plt.ylabel('Fitness')
plt.xlabel('Geração')
plt.legend(loc="lower right")
plt.show()


print(best_specimen_evolution[-1])
