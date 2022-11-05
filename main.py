#!/usr/bin/env python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


from model.my_model import NonParametricRegression
from functions.distances import euclidian_distance, manhattan_distance, chebyshev_distance
from functions.kernels import uniform_kernel, triangular_kernel, quartic_kernel
from functions.common import get_dataframe_from_arff

df = get_dataframe_from_arff("dataset/dataset_191_wine.arff")

dummies = pd.get_dummies(df['class'])
y = dummies.values

x = df.iloc[:, 1:]
# print(x)

x_scaled = MinMaxScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

model = NonParametricRegression(euclidian_distance, uniform_kernel, neighbours_count=3)

model.fit(x_train, y_train)
result = model.predict(x_test)
print(f'model.predict(x_test): \n {result}')

y_pred = np.argmax(result, axis = 1)
y_true = np.argmax(y_test, axis = 1)
# print(np.sqrt(len(x)))
# print(type(np.sqrt(len(x))))
# print(int(np.sqrt(len(x))))

print(accuracy_score(y_true, y_pred))

hyperparameters_w_t_uf = {'metrics': ['euclidean', 'manhattan', 'chebyshev'],
                          'kernels': ['uniform', 'triangular', 'epanechnikov', 'quartic'],
                          'n_neighbors': np.arange(1, (np.sqrt(len(x))).astype(int) + 1, 1)}


# # In[159]:
#
#
# def get_window_widths(X, metric='euclidean'):
#     distance_func_set = {'euclidean': nonparam_reg.euclidean_distance,
#                          'manhattan': nonparam_reg.manhattan_distance,
#                          'chebyshev': nonparam_reg.chebyshev_distance}
#     distance_func = distance_func_set[metric]
#
#     max_distances = []
#     for x1 in X:
#         max_distances.append(np.max([distance_func(x1, x2) for x2 in X]))
#     max_distance = max(max_distances)
#
#     return np.arange(max_distance / np.sqrt(len(X)), max_distance, max_distance / np.sqrt(len(X)))
#
#
# # In[160]:
#
#
# # hyperparameters_w_t_f = {'metrics' : ['euclidean', 'manhattan', 'chebyshev'],
# #                         'kernels' : ['uniform', 'triangular', 'epanechnikov', 'quartic'],
# #                         'window_widths' : [get_window_widths(X, metric) for metric in
# #                                            ['euclidean', 'manhattan', 'chebyshev']]}
# # hyperparameters_w_t_f = {'metrics' : ['euclidean', 'manhattan', 'chebyshev'],
# #                         'kernels' : ['uniform', 'triangular', 'epanechnikov', 'quartic'],
# #                         'window_widths' : get_window_widths(X, 'euclidean')}
# hyperparameters_w_t_f = {'metrics': ['euclidean', 'manhattan', 'chebyshev'],
#                          'kernels': ['uniform', 'triangular', 'epanechnikov', 'quartic']}
#
# # In[161]:
#
#
# hyperparameters_w_t_f
#
# # In[162]:
#
#
# hyperparameters_w_t_uf
#
#
# # ## Функция LeaveOneOut валидации
#
# # In[163]:
#
#
# def leave_one_out_validation(X, y, hyperparameters, window_type):
#     metric = hyperparameters['metric']
#     kernel = hyperparameters['kernel']
#
#     if window_type == 'fixed':
#         window_width = hyperparameters['window_width']
#         n_neighbors = 1
#     elif window_type == 'unfixed':
#         n_neighbors = hyperparameters['n_neighbors']
#         window_width = 1
#
#     y_pred = []
#     y_true = []
#     for i in range(0, len(X)):
#         X_train = np.delete(X, i, axis=0)
#         y_train = np.delete(y, i, axis=0)
#         X_test = X[i]
#         y_test = y[i]
#
#         model = nonparam_reg.Nonparam_reg(window_type, n_neighbors, window_width, metric, kernel)
#         model.fit(X_train, y_train)
#
#         y_pred.append(np.argmax(model.predict(X_test)))
#         y_true.append(np.argmax(y_test))
#
#     return f1_score(y_true, y_pred, average='macro')
#
#
# # ## Функция нахождения лучшей комбинации гиперпараметров
#
# # In[164]:
#
#
# def get_best_hyperparameters(X, y, dic_hyperparameters, window_type):
#     f_measure_max = 0
#     best_hyperparameters = 0
#     if window_type == 'fixed':
#         for metric in dic_hyperparameters['metrics']:
#             window_widths = get_window_widths(X, metric)
#             for kernel in dic_hyperparameters['kernels']:
#                 for window_width in window_widths:
#                     hyperparameters = {'metric': metric, 'kernel': kernel, 'window_width': window_width}
#                     f_measure_cur = leave_one_out_validation(X, y, hyperparameters, window_type)
#                     if f_measure_cur > f_measure_max:
#                         f_measure_max = f_measure_cur
#                         best_hyperparameters = hyperparameters
#     elif window_type == 'unfixed':
#         for metric in dic_hyperparameters['metrics']:
#             for kernel in dic_hyperparameters['kernels']:
#                 for n_neighbors in dic_hyperparameters['n_neighbors']:
#                     hyperparameters = {'metric': metric, 'kernel': kernel, 'n_neighbors': n_neighbors}
#                     f_measure_cur = leave_one_out_validation(X, y, hyperparameters, window_type)
#                     if f_measure_cur > f_measure_max:
#                         f_measure_max = f_measure_cur
#                         best_hyperparameters = hyperparameters
#
#     return f_measure_max, best_hyperparameters
#
#
# # ## Вычисление лучшей комбинации гиперпараметров
#
# # In[ ]:
#
#
# get_best_hyperparameters(X, y, hyperparameters_w_t_uf, 'unfixed')
#
# # In[ ]:
#
#
# get_best_hyperparameters(X, y, hyperparameters_w_t_f, 'fixed')
#
#
# # ## Вычисление зависимости F1-меры от числа ближайших соседей или ширины окна для лучшей комбинации гиперпараметров
#
# # In[ ]:
#
#
# def get_f1_scores(X, y, hyperparameters, window_type):
#     metric = hyperparameters['metric']
#     kernel = hyperparameters['kernel']
#
#     f1_scores = []
#     if window_type == 'fixed':
#         n_neighbors = 1
#         window_widths = get_window_widths(X, metric)
#         for window_width in window_widths:
#             y_pred = []
#             y_true = []
#             for i in range(0, len(X)):
#                 X_train = np.delete(X, i, axis=0)
#                 y_train = np.delete(y, i, axis=0)
#                 X_test = X[i]
#                 y_test = y[i]
#
#                 model = nonparam_reg.Nonparam_reg(window_type, n_neighbors, window_width, metric, kernel)
#                 model.fit(X_train, y_train)
#
#                 y_pred.append(np.argmax(model.predict(X_test)))
#                 y_true.append(np.argmax(y_test))
#
#             f1_scores.append((window_width, f1_score(y_true, y_pred, average='macro')))
#
#     elif window_type == 'unfixed':
#         window_width = 1
#         array_n_neighbors = np.arange(1, (np.sqrt(len(X))).astype(int) + 1, 1)
#         for n_neighbors in array_n_neighbors:
#             y_pred = []
#             y_true = []
#             for i in range(0, len(X)):
#                 X_train = np.delete(X, i, axis=0)
#                 y_train = np.delete(y, i, axis=0)
#                 X_test = X[i]
#                 y_test = y[i]
#
#                 model = nonparam_reg.Nonparam_reg(window_type, n_neighbors, window_width, metric, kernel)
#                 model.fit(X_train, y_train)
#
#                 y_pred.append(np.argmax(model.predict(X_test)))
#                 y_true.append(np.argmax(y_test))
#
#             f1_scores.append((n_neighbors, f1_score(y_true, y_pred, average='macro')))
#
#     return f1_scores
#
#
# # In[ ]:
#
#
# f1_scores_uf = get_f1_scores(x, y, {'metric': 'manhattan', 'kernel': 'uniform'}, 'unfixed')
#
# # In[ ]:
#
#
# f1_scores_uf
#
# # In[ ]:
#
#
# f1_scores_f = get_f1_scores(x, y, {'metric': 'manhattan', 'kernel': 'triangular'}, 'fixed')
#
# # In[ ]:
#
#
# f1_scores_f
#
# # In[ ]:
#
#
# # задаем размеры графика
# plt.figure(figsize=[14, 7])
#
# # даем название осям и графику
# plt.xlabel('число ближайших соседей', fontsize=14)
# plt.ylabel('F1-мера', fontsize=14)
# plt.title('График зависимости F1-меры от числа ближайших соседей', fontsize=15)
#
# # рисуем точки
# plt.plot([row[0] for row in f1_scores_uf], [row[1] for row in f1_scores_uf], '--b', label='F1-мера')
#
# # задаем легенду
# plt.legend(loc='lower right', fontsize=14)
#
# # устанавлием сетку
# plt.grid()
#
# # показываем график
# plt.show()
#
# # In[ ]:
#
#
# # задаем размеры графика
# plt.figure(figsize=[14, 7])
#
# # даем название осям и графику
# plt.xlabel('ширина окна', fontsize=14)
# plt.ylabel('F1-мера', fontsize=14)
# plt.title('График зависимости F1-меры от ширины окна', fontsize=15)
#
# # рисуем точки
# plt.plot([row[0] for row in f1_scores_f], [row[1] for row in f1_scores_f], '--b', label='F1-мера')
#
# # задаем легенду
# plt.legend(loc='upper right', fontsize=14)
#
# # устанавлием сетку
# plt.grid()
#
# # показываем график
# plt.show()
#
#
