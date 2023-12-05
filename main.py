import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import *


def euclidean_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += (x1[i] - x2[i]) ** 2
    return math.sqrt(dist)


def manhattan_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += math.fabs(x1[i] - x2[i])
    return dist


def chebyshev_distance(x1, x2):
    dist = list()
    for i in range(len(x1)):
        dist.append(math.fabs(x1[i] - x2[i]))
    return max(dist)


def uniform_kernel(u):
    if math.fabs(u) <= 1:
        return 0.5
    else:
        return 0


def triangular_kernel(u):
    if math.fabs(u) <= 1:
        return 1 - math.fabs(u)
    else:
        return 0


def epanechnikov_kernel(u):
    if math.fabs(u) <= 1:
        return 0.75 * (1 - u ** 2)
    else:
        return 0


def quartic_kernel(u):
    if math.fabs(u) <= 1:
        return 15 * (1 * u ** 2) ** 2 / 16
    else:
        return 0


def fixed_window_size_calc(element, points, dist_func, window_size_parameter):
    return window_size_parameter


def not_fixed_window_size_calc(element, points, dist_func, window_size_parameter):
    distances = list()
    for point in points:
        dist = dist_func(element, point)
        distances.append(dist)
    distances.sort()
    return distances[window_size_parameter - 1]


def find_max_dist(dist_calculation, points):
    p = points[0]
    idx = 0
    max_d = 0
    for i in range(len(points)):
        if dist_calculation(p, points[i]) > max_d:
            idx = i
            max_d = dist_calculation(p, points[i])
    p = points[idx]
    idx = 0
    max_d = 0
    for i in range(len(points)):
        if dist_calculation(p, points[i]) > max_d:
            idx = i
            max_d = dist_calculation(p, points[i])
    return dist_calculation(p, points[idx])


def classify(item, matrix_of_objects, matrix_of_labels, dist_func, ker_func, window_size):
    sum_yw = np.zeros(len(matrix_of_labels[0]))
    sum_w = 1e-10
    for i in range(len(matrix_of_objects)):
        sum_yw += matrix_of_labels[i] * ker_func(dist_func(matrix_of_objects[i], item) / window_size)
        sum_w += ker_func(dist_func(matrix_of_objects[i], item) / window_size)
    prediction = (1 / sum_w) * sum_yw
    max_component = 0
    max_comp_val = prediction[0]
    for i in range(len(prediction)):
        if prediction[i] > max_comp_val:
            max_comp_val = prediction[i]
            max_component = i
    prediction = np.zeros(len(prediction))
    prediction[max_component] = 1
    return prediction


def validation(matrix_of_objects, matrix_of_labels, dist_func, ker_func, window_size_calculation,
               window_size_parameter):
    size = len(matrix_of_labels[0])
    confusion_matrix = np.zeros((size, size), dtype=int)
    for i in range(len(matrix_of_objects)):
        item = matrix_of_objects[i]
        label = matrix_of_labels[i]
        cur_step_matrix_of_objects = np.delete(matrix_of_objects, i, axis=0)
        cur_step_matrix_of_labels = np.delete(matrix_of_labels, i, axis=0)
        window_size = window_size_calculation(item, cur_step_matrix_of_objects, dist_func, window_size_parameter)
        predicted_label = classify(item, cur_step_matrix_of_objects, cur_step_matrix_of_labels, dist_func, ker_func,
                                   window_size)
        predicted_label = predicted_label.astype(int)
        label = label.astype(int)
        actual_class = 0
        predicted_class = 0
        for j in range(size):
            if label[j] > 0:
                actual_class = j
            if predicted_label[j] > 0:
                predicted_class = j
        confusion_matrix[predicted_class][actual_class] += 1
    precision = 0
    recall = 0
    for i in range(size):
        precision += confusion_matrix[i][i] * np.sum(confusion_matrix, axis=0)[i] / (
                    np.sum(confusion_matrix, axis=1)[i] + 1e-10)
        recall += confusion_matrix[i][i]
    precision /= len(matrix_of_objects)
    recall /= len(matrix_of_objects)
    f_measure = 2 * precision * recall / (precision + recall + 1e-10)
    ans = (f_measure, dist_func, ker_func, window_size_calculation, window_size_parameter)
    return ans


def find_best_model(x, y):
    dist_calc = (euclidean_distance, manhattan_distance, chebyshev_distance)
    kernel = (uniform_kernel, triangular_kernel, epanechnikov_kernel, quartic_kernel)
    fixed_window_size = (True, False)
    models = list()
    i = 0
    for dist in dist_calc:
        for ker in kernel:
            for fixed in fixed_window_size:
                if fixed:
                    i += 1
                    print(i)
                    max_d = find_max_dist(dist, x)
                    distances = np.arange(max_d * 1.2 / math.sqrt(len(x)), max_d, max_d / math.sqrt(len(x)))
                    for distance in distances:
                        print(dist, ker, fixed_window_size_calc, distance)
                        models.append(validation(x, y, dist, ker, fixed_window_size_calc, distance))
                else:
                    i += 1
                    print(i)
                    distances = np.arange(1, math.sqrt(len(x)), 1, dtype=int)
                    for distance in distances:
                        print(dist, ker, not_fixed_window_size_calc, distance)
                        models.append(validation(x, y, dist, ker, not_fixed_window_size_calc, distance))

    models.sort(key=lambda tup: tup[0])
    models = models[::-1]
    return models[0:10]


def draw_plot(x, y):
    values = list(np.arange(1, math.sqrt(len(x)) * 2, 1, dtype=int))
    f_measure = list()
    print(values)
    for i in range(len(values)):
        f_measure.append(validation(x, y, manhattan_distance, quartic_kernel, not_fixed_window_size_calc, values[i])[0])
    print(f_measure)
    plt.plot(values, f_measure)
    plt.title("Зависимость F-меры от числа ближайших соседей")
    plt.xlabel("Число ближайших соседей")
    plt.ylabel("F-мера")
    plt.show()


dataset = pd.read_csv('data/wine.csv')
x = dataset.iloc[:, dataset.columns != 'class'].values
min_max = dataset_minmax(x)
normalize(x, min_max)  # x - матрица объекты-признаки

onehot_encoder = OneHotEncoder(sparse=False)
categorical_columns = dataset.columns[dataset.dtypes == 'object'].union(['class'])
encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(dataset[categorical_columns]))
y = encoded_categorical_columns.values  # y - метки

'''best_models = find_best_model(x, y)
print("\n ------------------------ \n")
for entry in best_models:
    print(entry)'''

draw_plot(x, y)
