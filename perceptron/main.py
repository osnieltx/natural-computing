import random
from typing import List, Tuple

import matplotlib.pyplot as plt


# Make a prediction with weights
def predict(a_input, perceptron):
    activation = sum(a * b for a, b in zip(a_input, perceptron['w'])) + perceptron['b']

    # Semi-linear activation
    if activation >= 1.:
        return 1.
    elif activation >= 0.:
        return activation
    return 0.


def train(train_data):
    perceptrons = [{'b': 0, 'w': [0.] * len(train_data[0])} for _ in range(len(train_data))]
    for epoch in range(n_epoch):
        all_error = 0.
        for p_i, perceptron in enumerate(perceptrons):
            sum_error = 0.
            for train_i in train_data:
                prediction = predict(train_i, perceptron)
                error = float(train_i == train_data[p_i]) - prediction
                sum_error += error ** 2
                # Updates bias and weights
                perceptron['b'] = perceptron['b'] + l_rate * error
                perceptron['w'] = [w + l_rate * error * v for w, v in zip(perceptron['w'], train_i)]

            print(f'epoch={epoch}, perceptron={p_i}, error={sum_error}')
            all_error += sum_error
        if not all_error:
            break
    print('--------------------------------------------------------')
    return perceptrons


def plot_weights():
    fig, axis = plt.subplots(2, len(train_data))
    for index, ax in enumerate(axis[0]):
        data_matrix = [train_data[index][i:i + 10] for i in range(0, 120, 10)]
        ax.matshow(data_matrix)
        ax.axis('off')
    for index, ax in enumerate(axis[1]):
        ax.matshow([perceptrons[index]['w'][i:i + 10] for i in range(0, 120, 10)])
        ax.axis('off')
    plt.show()



# Getting data
with open('char8_12x10.txt', 'r') as file:
    file.readline()
    rows = [[int(x) for x in line.split()] for line in file]
    train_data: List[List[int]] = [list(col) for col in zip(*rows)]

# Generates test data with noise
test_data: List[Tuple] = []
for noise_prcnt in range(101):
    new_test_data = []
    for pattern in train_data:
        new_pattern = []
        indexes_to_add_noise = random.sample(range(121), noise_prcnt*120//100)
        for index, pixel in enumerate(pattern):
            if index in indexes_to_add_noise:
                new_pattern.append(-1 if pixel == 1 else 1)
            else:
                new_pattern.append(pixel)
        new_test_data.append(new_pattern)
    test_data.append((noise_prcnt, new_test_data))

l_rate = .001
n_epoch = 100
perceptrons = train(train_data)

# Tests with the noise data
results = []
for noise_volumn, noise_patterns in test_data:
    results.append((8-sum(abs(1 - predict(noise_patterns[p_i], perceptron))
                   for p_i, perceptron in enumerate(perceptrons)))/8*100)

# plt.plot(range(101), results, 'r', label='0.1')

plot_weights()
#
# l_rate = .3
# perceptrons = train(train_data)
# results = []
# for noise_volumn, noise_patterns in test_data:
#     results.append((8 - sum(abs(1 - predict(noise_patterns[p_i], perceptron))
#                             for p_i, perceptron in enumerate(perceptrons))) / 8 * 100)
# plt.plot(range(101), results, 'g', label='0.3')
#
#
# l_rate = .001
# perceptrons = train(train_data)
# results = []
# for noise_volumn, noise_patterns in test_data:
#     results.append((8 - sum(abs(1 - predict(noise_patterns[p_i], perceptron))
#                             for p_i, perceptron in enumerate(perceptrons))) / 8 * 100)
# plt.plot(range(101), results, 'b', label='0.001')
#
#
# l_rate = .0001
# perceptrons = train(train_data)
# results = []
# for noise_volumn, noise_patterns in test_data:
#     results.append((8 - sum(abs(1 - predict(noise_patterns[p_i], perceptron))
#                             for p_i, perceptron in enumerate(perceptrons))) / 8 * 100)
# plt.plot(range(101), results, 'y', label='0.0001')
#
#
# l_rate = .00001
# perceptrons = train(train_data)
# results = []
# for noise_volumn, noise_patterns in test_data:
#     results.append((8 - sum(abs(1 - predict(noise_patterns[p_i], perceptron))
#                             for p_i, perceptron in enumerate(perceptrons))) / 8 * 100)
# plt.plot(range(101), results, 'purple', label='0.00001')
#
# plt.ylabel('Acurácia (%)')
# plt.xlabel('Ruído (%)')
# plt.title('Performance de diferentes taxas de aprendizagem')
# plt.legend(loc='lower left')
# plt.grid(True)
# plt.show()
