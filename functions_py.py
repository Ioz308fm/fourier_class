import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from numba import njit
# Все функции

from numba import njit
import numpy as np
import pandas as pd

# functions_py.py (новые оптимизированные функции с префиксами qwen_)
from numba import njit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import optuna

# Оптимизированная функция вычисления коэффициентов Фурье
@njit
def compute_coefficients(g_values, D, M):
    """Вычисление коэффициентов Фурье с использованием Numba"""
    row_labels = np.zeros(2*M - 1, dtype=np.int32)
    for idx in range(1, M):
        row_labels[2*idx-1] = idx
        row_labels[2*idx] = -idx
    coef_values = np.zeros(len(row_labels), dtype=np.complex128)
    for k, idx in enumerate(row_labels):
        sum_coef = 0.0 + 0.0j
        if idx == 0:
            sum_coef = np.sum(g_values) / D
        else:
            for j in range(D):
                g = g_values[j]
                a = 2*np.pi * j / D
                b = 2*np.pi * (j + 1) / D
                left = np.exp(-1j * b * idx)
                right = np.exp(-1j * a * idx)
                sum_coef += g * (left - right) / (-1j * idx)
            sum_coef /= 2*np.pi
        coef_values[k] = sum_coef
    return coef_values, row_labels

# Оптимизированная функция расчета коэффициентов Фурье
def qwen_optimized_fourier_coef_calc(target, data_train, labeled_piece_func, D, M):
    list_of_labels = np.sort(data_train[target].unique())
    dim = (data_train.shape[1] - 1) // 2
    row_labels = np.zeros(2*M - 1, dtype=int)
    
    for idx in range(1, M):
        row_labels[2*idx-1] = idx
        row_labels[2*idx] = -idx
        
    labeled_coef = []
    for i in range(len(list_of_labels)):
        dim_coef = []
        for p in range(dim):
            g_values = np.array([labeled_piece_func[i][p][j][1] for j in range(D)])
            coef_values, _ = compute_coefficients(g_values, D, M)
            
            coef_df = pd.DataFrame({'value': coef_values}, index=row_labels)
            dim_coef.append(coef_df)
        labeled_coef.append(dim_coef)
    return labeled_coef

# Оптимизированная функция определения кусочно-постоянной функции
def qwen_optimized_func_def(target, data_train, polar_data, D):
    list_of_labels = np.sort(data_train[target].unique())
    Dlist = np.linspace(0, 2*np.pi, D+1)
    dim = polar_data[0].shape[1]
    labeled_func = []
    
    for i in range(len(list_of_labels)):
        piece_wise_func = []
        current_polar = polar_data[i]
        for p in range(dim // 2):
            angles = current_polar.iloc[:, 2*p].values
            radii = current_polar.iloc[:, 2*p+1].values
            
            bins = np.digitize(angles, Dlist) - 1
            bins[bins == D] = D - 1
            
            max_values = np.full(D, np.nan)
            for j in range(D):
                mask = (bins == j)
                if np.any(mask):
                    max_values[j] = np.max(radii[mask])
            
            temp = np.empty((D, 2))
            temp[:, 0] = Dlist[:-1]
            temp[:, 1] = max_values
            
            # Заполнение пропусков
            for j in range(D):
                if np.isnan(temp[j, 1]):
                    k = 1
                    while k <= D:
                        left = (j - k) % D
                        right = (j + k) % D
                        if not np.isnan(temp[left, 1]) and not np.isnan(temp[right, 1]):
                            temp[j, 1] = (temp[left, 1] + temp[right, 1]) / 2
                            break
                        elif not np.isnan(temp[left, 1]):
                            temp[j, 1] = temp[left, 1]
                            break
                        elif not np.isnan(temp[right, 1]):
                            temp[j, 1] = temp[right, 1]
                            break
                        else:
                            k += 1
            piece_wise_func.append(temp)
        labeled_func.append(piece_wise_func)
    return labeled_func

# Оптимизированный пайплайн обучения
def qwen_pipeline_train(data_with_labels, target, state=42, conversion_state=True, write_state=False, D=10, M=3):
    data_train, data_test = train_test_split(data_with_labels, stratify=data_with_labels[target], test_size=0.33, random_state=state)
    data_train_means = data_train.groupby(target).mean()
    
    if conversion_state:
        polar_data = qwen_polar_conversion_fast(target, data_train, data_train_means)
        if write_state:
            for i in range(10):
                polar_data[i].to_csv(f"polar_converted_{i}.csv")
    else:
        polar_data = [pd.read_csv(f"polar_converted_{i}.csv") for i in range(10)]
    
    labeled_piece_func = qwen_optimized_func_def(target, data_train, polar_data, D)
    fourier_coef = qwen_optimized_fourier_coef_calc(target, data_train, labeled_piece_func, D, M)
    return data_train, data_test, data_train_means, fourier_coef

# Оптимизированный тестовый алгоритм
def qwen_optimized_test_class_alg(prefix, file_name, target, data_test, data_train, data_train_means, fourier_coef, tmp_test_coef, rewrite=True):
    log_dir = os.path.join('logs', prefix)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'log_{file_name}.txt')
    
    if rewrite:
        with open(log_file_path, 'w') as f:
            f.write('')
            
    list_of_labels = np.sort(data_train[target].unique())
    dim = (data_train.shape[1] - 1) // 2
    num_classes = len(list_of_labels)
    image_data = data_test.drop(columns=[target]).values
    real_labels = data_test[target].values
    
    all_means = np.array([data_train_means.loc[label].values for label in list_of_labels])
    all_votes = np.zeros((image_data.shape[0], num_classes), dtype=int)
    
    for i, current_label in enumerate(list_of_labels):
        biased = image_data - all_means[i]
        x = biased[:, ::2]
        y = biased[:, 1::2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) % (2*np.pi)
        
        for j in range(dim):
            current_phi = phi[:, j]
            current_r = r[:, j]
            coefs = fourier_coef[i][j]['value'].values
            indexes = fourier_coef[i][j].index.values
            
            real_part = np.sum(
                np.cos(np.outer(current_phi, indexes)) * coefs.real - 
                np.sin(np.outer(current_phi, indexes)) * coefs.imag,
                axis=1
            )
            all_votes[:, i] += (current_r <= tmp_test_coef * real_part).astype(int)
    
    predicted_labels = np.argmax(all_votes, axis=1)
    
    true_positive = np.zeros(num_classes, dtype=int)
    false_positive = np.zeros(num_classes, dtype=int)
    true_negative = np.zeros(num_classes, dtype=int)
    false_negative = np.zeros(num_classes, dtype=int)
    
    for C in range(num_classes):
        true_positive[C] = np.sum((real_labels == C) & (predicted_labels == C))
        false_positive[C] = np.sum((real_labels != C) & (predicted_labels == C))
        true_negative[C] = np.sum((real_labels != C) & (predicted_labels != C))
        false_negative[C] = np.sum((real_labels == C) & (predicted_labels != C))
    
    with open(log_file_path, 'a') as f:
        print(f'Total images processed: {len(real_labels)}', file=f)
        print(f'true_positive = {true_positive.tolist()}', file=f)
        print(f'false_positive = {false_positive.tolist()}', file=f)
        print(f'true_negative = {true_negative.tolist()}', file=f)
        print(f'false_negative = {false_negative.tolist()}', file=f)
        
        TP = np.sum(true_positive)
        precision_arr = true_positive / (true_positive + false_positive + 1e-10)
        accuracy = TP / len(real_labels)
        precision = np.mean(precision_arr)
        
        print(f'TP = {TP.sum()}', file=f)
        print(f'accuracy = {accuracy}', file=f)
        print(f'precision = {precision}', file=f)
    
    return accuracy

# Оптимизированное преобразование в полярные координаты
def qwen_polar_conversion_fast(target, data_train, data_train_means):
    list_of_labels = np.sort(data_train[target].unique())
    data = [data_train[data_train[target] == label].copy() for label in list_of_labels]
    polar_data = []
    
    for idx, df in enumerate(data):
        current_label = df[target].iloc[0]
        data_unlabeled = df.drop(columns=[target]).values
        means = data_train_means.loc[current_label].values
        
        biased = data_unlabeled - means
        x = biased[:, ::2]
        y = biased[:, 1::2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) % (2*np.pi)
        
        polar = np.empty_like(biased)
        polar[:, ::2] = phi
        polar[:, 1::2] = r
        polar_data.append(pd.DataFrame(polar, columns=df.drop(columns=[target]).columns))
    
    return polar_data

# Гиперпараметрическая оптимизация с Optuna
def run_hyperparameter_optimization(pred_obr_func, input_data, target, conversion_state, write_state, state, base, prefix, D_left, D_right, M_left, M_right, T_left, T_right, max_iter):
    """Запуск гиперпараметрической оптимизации"""
    # Параметры оптимизации
    # input_data = pd.read_csv('train.csv')
    # base = ''
    # target = 'label'
    # prefix = 'COEF_TEST_1.05_column_row_'
    # D_left = 2800
    # D_right = 2821
    # M_left = 2
    # M_right = 11
    # t_t = [1 + i/100 for i in range(11)]
    res_file_path = os.path.join('logs', f'results_{prefix}_optuna.txt')

    # Целевая функция для оптимизации
    def objective(trial):
        """Целевая функция для оптимизации"""
        # Пространство параметров
        D_val = trial.suggest_int('D', D_left, D_right-1)
        M_val = trial.suggest_int('M', M_left, M_right-1)
        t_coef = trial.suggest_float('t', T_left, T_right)
        
        print(f'Тестируем параметры: D={D_val}, M={M_val}, t={t_coef}')
        
        # Обработка данных
        data_with_labels = pred_obr_func(input_data)
        
        # Обучение модели
        data_train, data_test, data_train_means, fourier_coef = qwen_pipeline_train(
            data_with_labels, target, state, conversion_state, write_state, D_val, M_val
        )
        
        # Тестирование
        accuracy = qwen_optimized_test_class_alg(
            prefix, f'D{D_val}M{M_val}', target, data_test, data_train, 
            data_train_means, fourier_coef, t_coef, rewrite=True
        )
        
        return accuracy

    # Создание и запуск исследования
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=max_iter)  # 50 итераций оптимизации

    # Сохранение результатов
    best_params = study.best_params
    best_value = study.best_value

    with open(res_file_path, 'a') as f:
        print(f'Лучшие параметры: {best_params}', file=f)
        print(f'Лучшая точность: {best_value}', file=f)
    
    return best_params, best_value


# QWEN 11111
# @njit
# def compute_coefficients(g_values, D, M):
#     """Вычисление коэффициентов Фурье с использованием Numba"""
#     row_labels = np.zeros(2*M - 1, dtype=np.int32)
#     for idx in range(1, M):
#         row_labels[2*idx-1] = idx
#         row_labels[2*idx] = -idx
#     coef_values = np.zeros(len(row_labels), dtype=np.complex128)
#     for k, idx in enumerate(row_labels):
#         sum_coef = 0.0 + 0.0j
#         if idx == 0:
#             sum_coef = np.sum(g_values) / D
#         else:
#             for j in range(D):
#                 g = g_values[j]
#                 a = 2*np.pi * j / D
#                 b = 2*np.pi * (j + 1) / D
#                 left = np.exp(-1j * b * idx)
#                 right = np.exp(-1j * a * idx)
#                 sum_coef += g * (left - right) / (-1j * idx)
#             sum_coef /= 2*np.pi
#         coef_values[k] = sum_coef
#     return coef_values, row_labels

# def qwen_optimized_fourier_coef_calc(target, data_train, labeled_piece_func, D, M):
#     list_of_labels = np.sort(data_train[target].unique())
#     dim = (data_train.shape[1] - 1) // 2
#     row_labels = np.zeros(2*M - 1, dtype=int)
    
#     for idx in range(1, M):
#         row_labels[2*idx-1] = idx
#         row_labels[2*idx] = -idx
        
#     labeled_coef = []
#     for i in range(len(list_of_labels)):
#         dim_coef = []
#         for p in range(dim):
#             g_values = np.array([labeled_piece_func[i][p][j][1] for j in range(D)])
#             coef_values, _ = compute_coefficients(g_values, D, M)
            
#             coef_df = pd.DataFrame({'value': coef_values}, index=row_labels)
#             dim_coef.append(coef_df)
#         labeled_coef.append(dim_coef)
#     return labeled_coef

# def qwen_optimized_func_def(target, data_train, polar_data, D):
#     list_of_labels = np.sort(data_train[target].unique())
#     Dlist = np.linspace(0, 2*np.pi, D+1)
#     dim = polar_data[0].shape[1]
#     labeled_func = []
    
#     for i in range(len(list_of_labels)):
#         piece_wise_func = []
#         current_polar = polar_data[i]
#         for p in range(dim // 2):
#             angles = current_polar.iloc[:, 2*p].values
#             radii = current_polar.iloc[:, 2*p+1].values
            
#             bins = np.digitize(angles, Dlist) - 1
#             bins[bins == D] = D - 1
            
#             max_values = np.full(D, np.nan)
#             for j in range(D):
#                 mask = (bins == j)
#                 if np.any(mask):
#                     max_values[j] = np.max(radii[mask])
            
#             temp = np.empty((D, 2))
#             temp[:, 0] = Dlist[:-1]
#             temp[:, 1] = max_values
            
#             # Заполнение пропусков
#             for j in range(D):
#                 if np.isnan(temp[j, 1]):
#                     k = 1
#                     while k <= D:
#                         left = (j - k) % D
#                         right = (j + k) % D
#                         if not np.isnan(temp[left, 1]) and not np.isnan(temp[right, 1]):
#                             temp[j, 1] = (temp[left, 1] + temp[right, 1]) / 2
#                             break
#                         elif not np.isnan(temp[left, 1]):
#                             temp[j, 1] = temp[left, 1]
#                             break
#                         elif not np.isnan(temp[right, 1]):
#                             temp[j, 1] = temp[right, 1]
#                             break
#                         else:
#                             k += 1
#             piece_wise_func.append(temp)
#         labeled_func.append(piece_wise_func)
#     return labeled_func

# def qwen_pipeline_train(data_with_labels, target, state = 42, conversion_state = True, write_state = False, D = 10, M = 3):
#     data_train, data_test = train_test_split(data_with_labels, stratify=data_with_labels[target], test_size=0.33, random_state=state)
#     data_train_means = data_train.groupby(target).mean()
    
#     if conversion_state:
#         polar_data = qwen_polar_conversion_fast(target, data_train, data_train_means)
#         if write_state:
#             for i in range(10):
#                 polar_data[i].to_csv(f"polar_converted_{i}.csv")
#     else:
#         polar_data = [pd.read_csv(f"polar_converted_{i}.csv") for i in range(10)]
    
#     labeled_piece_func = qwen_optimized_func_def(target, data_train, polar_data, D)
#     fourier_coef = qwen_optimized_fourier_coef_calc(target, data_train, labeled_piece_func, D, M)
#     return data_train, data_test, data_train_means, fourier_coef

# def qwen_optimized_test_class_alg(prefix, file_name, target, data_test, data_train, data_train_means, fourier_coef, tmp_test_coef, rewrite=True):
#     log_dir = os.path.join('logs', prefix)
#     os.makedirs(log_dir, exist_ok=True)
#     log_file_path = os.path.join(log_dir, f'log_{file_name}.txt')
    
#     if rewrite:
#         with open(log_file_path, 'w') as f:
#             f.write('')
            
#     list_of_labels = np.sort(data_train[target].unique())
#     dim = (data_train.shape[1] - 1) // 2
#     num_classes = len(list_of_labels)
#     image_data = data_test.drop(columns=[target]).values
#     real_labels = data_test[target].values
    
#     all_means = np.array([data_train_means.loc[label].values for label in list_of_labels])
#     all_votes = np.zeros((image_data.shape[0], num_classes), dtype=int)
    
#     for i, current_label in enumerate(list_of_labels):
#         biased = image_data - all_means[i]
#         x = biased[:, ::2]
#         y = biased[:, 1::2]
#         r = np.sqrt(x**2 + y**2)
#         phi = np.arctan2(y, x) % (2*np.pi)
        
#         for j in range(dim):
#             current_phi = phi[:, j]
#             current_r = r[:, j]
#             coefs = fourier_coef[i][j]['value'].values
#             indexes = fourier_coef[i][j].index.values
            
#             real_part = np.sum(
#                 np.cos(np.outer(current_phi, indexes)) * coefs.real - 
#                 np.sin(np.outer(current_phi, indexes)) * coefs.imag,
#                 axis=1
#             )
#             all_votes[:, i] += (current_r <= tmp_test_coef * real_part).astype(int)
    
#     predicted_labels = np.argmax(all_votes, axis=1)
    
#     true_positive = np.zeros(num_classes, dtype=int)
#     false_positive = np.zeros(num_classes, dtype=int)
#     true_negative = np.zeros(num_classes, dtype=int)
#     false_negative = np.zeros(num_classes, dtype=int)
    
#     for C in range(num_classes):
#         true_positive[C] = np.sum((real_labels == C) & (predicted_labels == C))
#         false_positive[C] = np.sum((real_labels != C) & (predicted_labels == C))
#         true_negative[C] = np.sum((real_labels != C) & (predicted_labels != C))
#         false_negative[C] = np.sum((real_labels == C) & (predicted_labels != C))
    
#     with open(log_file_path, 'a') as f:
#         print(f'Total images processed: {len(real_labels)}', file=f)
#         print(f'true_positive = {true_positive.tolist()}', file=f)
#         print(f'false_positive = {false_positive.tolist()}', file=f)
#         print(f'true_negative = {true_negative.tolist()}', file=f)
#         print(f'false_negative = {false_negative.tolist()}', file=f)
        
#         TP = np.sum(true_positive)
#         precision_arr = true_positive / (true_positive + false_positive + 1e-10)
#         accuracy = TP / len(real_labels)
#         precision = np.mean(precision_arr)
        
#         print(f'TP = {TP.sum()}', file=f)
#         print(f'accuracy = {accuracy}', file=f)
#         print(f'precision = {precision}', file=f)
    
#     return accuracy

# def qwen_polar_conversion_fast(target, data_train, data_train_means):
#     list_of_labels = np.sort(data_train[target].unique())
#     data = [data_train[data_train[target] == label].copy() for label in list_of_labels]
#     polar_data = []
    
#     for idx, df in enumerate(data):
#         current_label = df[target].iloc[0]
#         data_unlabeled = df.drop(columns=[target]).values
#         means = data_train_means.loc[current_label].values
        
#         biased = data_unlabeled - means
#         x = biased[:, ::2]
#         y = biased[:, 1::2]
#         r = np.sqrt(x**2 + y**2)
#         phi = np.arctan2(y, x) % (2*np.pi)
        
#         polar = np.empty_like(biased)
#         polar[:, ::2] = phi
#         polar[:, 1::2] = r
#         polar_data.append(pd.DataFrame(polar, columns=df.drop(columns=[target]).columns))
    
#     return polar_data



# # # # # # # # # # # # # # # # # # # # # # # #
# Тренировка пайплайн
def pipeline_train(data_with_labels, target, state = 42, conversion_state = True, write_state = False, D = 10, M = 3):
    
    # разбиваем данные на тренировочные и тестовые
    data_train, data_test = train_test_split(data_with_labels, stratify=data_with_labels[target], test_size=0.33, random_state=state)
    # средние по всем координатам, для всех таргетов
    data_train_means = data_train.groupby(target).mean()

    # print('——— Converting to polar ———')
    
    # перевод в полярные координаты
    if conversion_state:
        polar_data = polar_conversion_fast(target, data_train, data_train_means)
        # сохраним данные в полярных координатах
        if write_state:
            # print('——— Saving converted values ———')
            for i in range (10):
                polar_data[i].to_csv("polar_converted_" + str(i) + ".csv")
    else:
        polar_data = []
        for i in range (10):
            polar_data.append(pd.read_csv("polar_converted_" + str(i) + ".csv"))
    # результат в виде списка result[метка][индексы от 0 до 392][разбиение по полярному углу]
    # print('——— Calculating piece-wise function ———')
    labeled_piece_func = optimized_func_def(target, data_train, polar_data, D)
    # коэффициенты фурье
    # print('——— Calculating fourier coefficients ———')
    fourier_coef = optimized_fourier_coef_calc(target, data_train, labeled_piece_func, D, M)

    # print('——— Done ! ———')
    return data_train, data_test, data_train_means, fourier_coef
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
def optimized_fourier_coef_calc(target, data_train, labeled_piece_func, D, M):
    list_of_labels = np.sort(data_train[target].unique())
    dim = (data_train.shape[1] - 1) // 2
    
    # Создаем индексы коэффициентов Фурье [0, 1, -1, 2, -2, ...]
    row_labels = np.zeros(2*M - 1, dtype=int)
    for idx in range(1, M):
        row_labels[2*idx-1] = idx
        row_labels[2*idx] = -idx
    
    labeled_coef = []
    
    for i in range(len(list_of_labels)):
        dim_coef = []
        for p in range(dim):
            # Используем numpy array для хранения комплексных коэффициентов
            coef_values = np.zeros(len(row_labels), dtype=np.complex128)
            
            for k, idx in enumerate(row_labels):
                sum_coef = complex(0)
                
                if idx == 0:
                    # Для нулевого коэффициента - просто среднее значение
                    for j in range(D):
                        g = labeled_piece_func[i][p][j][1]  # Доступ к numpy массиву
                        sum_coef += g
                    sum_coef /= D
                else:
                    # Для ненулевых коэффициентов
                    for j in range(D):
                        g = labeled_piece_func[i][p][j][1]  # Доступ к numpy массиву
                        a = 2*np.pi * j / D
                        b = 2*np.pi * (j + 1) / D
                        
                        # Векторизованные вычисления
                        left = np.exp(-1j * b * idx)
                        right = np.exp(-1j * a * idx)
                        sum_coef += g * (left - right) / (-1j * idx)
                    
                    sum_coef /= 2*np.pi
                
                coef_values[k] = sum_coef
            
            # Создаем DataFrame с результатами
            coef_df = pd.DataFrame({'value': coef_values}, index=row_labels)
            dim_coef.append(coef_df)
        
        labeled_coef.append(dim_coef)
    
    return labeled_coef

# Дополнительная оптимизация с Numba (если возможно)
@njit
def compute_coefficients(g_values, D, M):
    """Вычисление коэффициентов Фурье с использованием Numba"""
    row_labels = np.zeros(2*M - 1, dtype=np.int32)
    for idx in range(1, M):
        row_labels[2*idx-1] = idx
        row_labels[2*idx] = -idx
    
    coef_values = np.zeros(len(row_labels), dtype=np.complex128)
    
    for k, idx in enumerate(row_labels):
        sum_coef = 0.0 + 0.0j
        
        if idx == 0:
            sum_coef = np.sum(g_values) / D
        else:
            for j in range(D):
                g = g_values[j]
                a = 2*np.pi * j / D
                b = 2*np.pi * (j + 1) / D
                
                left = np.exp(-1j * b * idx)
                right = np.exp(-1j * a * idx)
                sum_coef += g * (left - right) / (-1j * idx)
            
            sum_coef /= 2*np.pi
        
        coef_values[k] = sum_coef
    
    return coef_values, row_labels
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
def optimized_func_def(target, data_train, polar_data, D):
    list_of_labels = np.sort(data_train[target].unique())
    Dlist = np.linspace(0, 2*np.pi, D+1)
    dim = polar_data[0].shape[1]
    labeled_func = []
    
    for i in range(len(list_of_labels)):
        piece_wise_func = []
        current_polar = polar_data[i]
        
        for p in range(dim // 2):
            # Извлекаем нужные колонки
            angles = current_polar.iloc[:, 2*p].values
            radii = current_polar.iloc[:, 2*p+1].values
            
            # Векторизованное вычисление максимумов
            func_dim = np.empty((D, 2))
            func_dim[:, 0] = Dlist[:-1]  # Углы
            
            # Используем digitize для быстрого определения интервалов
            bins = np.digitize(angles, Dlist) - 1
            bins[bins == D] = D - 1  # Последний интервал включает правую границу
            
            # Вычисляем максимумы для каждого интервала
            for j in range(D):
                mask = (bins == j)
                if np.any(mask):
                    func_dim[j, 1] = np.max(radii[mask])
                else:
                    func_dim[j, 1] = np.nan
            
            # Восстановление пропущенных значений (оптимизированная версия)
            temp = func_dim.copy()
            temp = fill_missing_values(temp, D)
            
            piece_wise_func.append(temp)
        labeled_func.append(piece_wise_func)
    
    return labeled_func

@njit
def fill_missing_values(arr, D):
    """Точная реплика оригинальной логики заполнения пропусков"""
    for j in range(D):
        if np.isnan(arr[j, 1]):
            k = 1
            while k <= D:
                left = (j - k) % D  # Фикс: убрано избыточное +D
                right = (j + k) % D
                
                left_val = arr[left, 1]
                right_val = arr[right, 1]
                
                if not np.isnan(left_val) and not np.isnan(right_val):
                    arr[j, 1] = (left_val + right_val) / 2
                    break
                elif not np.isnan(left_val):
                    arr[j, 1] = left_val
                    break
                elif not np.isnan(right_val):
                    arr[j, 1] = right_val
                    break
                else:
                    k += 1
    return arr
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# тестирование и создание логов
def batch_get_labels(target, data_train, data_train_means, fourier_coef, test_images, tmp_test_coef):
    list_of_labels = np.sort(data_train[target].unique())
    dim = (data_train.shape[1] - 1) // 2
    num_classes = len(list_of_labels)
    num_images = test_images.shape[0]
    
    # Подготовка данных
    image_data = test_images.drop(columns=[target]).values
    real_labels = test_images[target].values
    
    # Предварительная обработка means
    all_means = np.array([data_train_means.loc[label].values for label in list_of_labels])
    
    # Инициализация массива для голосов
    all_votes = np.zeros((num_images, num_classes), dtype=int)
    
    for i, current_label in enumerate(list_of_labels):
        # Векторизованное смещение
        biased = image_data - all_means[i]
        
        # Полярные координаты для всех изображений сразу
        x = biased[:, ::2]
        y = biased[:, 1::2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) % (2*np.pi)
        
        # Подсчет голосов для всех изображений и всех пар координат
        for j in range(dim):
            current_phi = phi[:, j]
            current_r = r[:, j]
            
            # Получаем коэффициенты Фурье
            coefs = fourier_coef[i][j]['value'].values
            indexes = fourier_coef[i][j].index.values
            
            # Векторизованное вычисление ряда Фурье
            real_part = np.sum(
                np.cos(indexes * current_phi[:, None]) * coefs.real - 
                np.sin(indexes * current_phi[:, None]) * coefs.imag,
                axis=1
            )
            
            all_votes[:, i] += (current_r <= tmp_test_coef * real_part).astype(int)
    
    predicted_labels = np.argmax(all_votes, axis=1)
    return predicted_labels, real_labels

def optimized_test_class_alg(prefix, file_name, target, data_test, data_train, data_train_means, fourier_coef, tmp_test_coef, rewrite=True):
    # Создаем папку для логов, если её нет
    log_dir = os.path.join('logs', prefix)  # Папка logs/ваше_название
    os.makedirs(log_dir, exist_ok=True)  # Создаст папку, если её нет
    # Полный путь к файлу лога
    log_file_path = os.path.join(log_dir, f'log_{file_name}.txt')
    if rewrite:
        with open(log_file_path, 'w') as f:
            f.write('') 

    classes = np.sort(data_train[target].unique())
    num_classes = len(classes)
    
    # Пакетная обработка всех изображений
    predicted_labels, real_labels = batch_get_labels(
        target, data_train, data_train_means, fourier_coef, data_test, tmp_test_coef
    )
    
    # Инициализация счетчиков
    true_positive = np.zeros(num_classes, dtype=int)
    false_positive = np.zeros(num_classes, dtype=int)
    true_negative = np.zeros(num_classes, dtype=int)
    false_negative = np.zeros(num_classes, dtype=int)
    
    # Подсчет метрик
    for C in range(num_classes):
        true_positive[C] = np.sum((real_labels == C) & (predicted_labels == C))
        false_positive[C] = np.sum((real_labels != C) & (predicted_labels == C))
        true_negative[C] = np.sum((real_labels != C) & (predicted_labels != C))
        false_negative[C] = np.sum((real_labels == C) & (predicted_labels != C))
    
    # Логирование результатов
    with open(log_file_path, 'a') as f:
        print(f'Total images processed: {len(real_labels)}', file=f)
        print(f'true_positive = {true_positive.tolist()}', file=f)
        print(f'false_positive = {false_positive.tolist()}', file=f)
        print(f'true_negative = {true_negative.tolist()}', file=f)
        print(f'false_negative = {false_negative.tolist()}', file=f)
        
        TP = np.sum(true_positive)
        FP = np.sum(false_positive)
        precision_arr = np.zeros(num_classes)
        for C in range(num_classes):
            denominator = true_positive[C] + false_positive[C]
            precision_arr[C] = true_positive[C] / denominator if denominator != 0 else 0
        
        accuracy = TP / len(real_labels)
        precision = np.mean(precision_arr)

        print(f'TP = {TP.sum()}', file=f)
        print(f'accuracy = {accuracy}', file=f)
        print(f'precision = {precision}', file=f)
    
    # print('——— Testing is over ———')
    return accuracy
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# Классификация 1 объекта
def get_label_fast(target, data_train, data_train_means, fourier_coef, image):
    list_of_labels = data_train[target].unique()
    list_of_labels.sort()
    dim = (data_train.shape[1] - 1) // 2
    votes = np.zeros(len(list_of_labels), dtype=int)
    
    # Предварительно преобразуем image в numpy array
    image_data = image.drop(columns=[target]).values[0]  # Берем первую (и единственную) строку
    
    for i, current_label in enumerate(list_of_labels):
        # Сдвигаем на среднее (векторизованная операция)
        means = data_train_means.loc[current_label].values
        biased = image_data - means
        
        # Полярные координаты для всех пар сразу
        x = biased[::2]
        y = biased[1::2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) % (2*np.pi)
        
        # Подсчет голосов
        for j in range(dim):
            current_phi = phi[j]
            current_r = r[j]
            
            # Получаем коэффициенты Фурье как numpy массив
            coefs = fourier_coef[i][j]['value'].values
            indexes = fourier_coef[i][j].index.values
            
            # Вычисление ряда Фурье (оптимизированная версия)
            real_part = np.sum(np.cos(indexes * current_phi) * coefs.real - 
                        np.sin(indexes * current_phi) * coefs.imag)
            
            if current_r <= real_part:
                votes[i] += 1
                
    return np.argmax(votes)
# # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # #
# Преобразование в полярные координаты
def polar_conversion_fast(target, data_train, data_train_means):
    list_of_labels = np.sort(data_train[target].unique())
    
    # Подготовка данных - более эффективный способ
    data = [data_train[data_train[target] == label].copy() for label in list_of_labels]
    
    polar_data = []
    for idx, df in enumerate(data):
        current_label = df[target].iloc[0]
        # print(current_label)
        
        # Получаем данные без метки и средние значения
        data_unlabeled = df.drop(columns=[target]).values
        means = data_train_means.loc[current_label].values
        
        # Векторизованное вычисление смещенных координат
        biased = data_unlabeled - means
        
        # Подготовка массива для полярных координат
        n_pairs = biased.shape[1] // 2
        polar = np.empty_like(biased)
        
        # Векторизованный перевод в полярные координаты
        x = biased[:, ::2]  # Все x координаты
        y = biased[:, 1::2]  # Все y координаты
        
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        phi = phi % (2*np.pi)
        
        polar[:, ::2] = phi
        polar[:, 1::2] = r
        
        polar_data.append(pd.DataFrame(polar, columns=df.drop(columns=[target]).columns))
    
    return polar_data
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# перевод в полярные координаты
def convert_to_polar(row):
    row = row.copy()
    x = row.iloc[0]
    y = row.iloc[1]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)  # Автоматически обрабатывает все случаи
    phi = phi % (2*np.pi)   # Приводим к диапазону [0, 2π)
    row[row.index[0]] = phi
    row[row.index[1]] = r
    return row

def polar_conversion(target, data_train, data_train_means):
  list_of_labels = data_train[target].unique()
  list_of_labels.sort()
  #данные в виде списка, где каждый элемент это датафрейм содержащий лишь данные с определенной меткой
  data = []
  for labels in list_of_labels:
    data.append(data_train.loc[data_train[target] == labels].copy())
  #для каждого класса
  polar_data = []
  for idx in range(len(data)):
    #текущая метка
    current_label = data[idx][target].unique()[0]
    # print(current_label)
    #данные без метки
    data_unlabeled = data[idx].drop(columns=[target])
    #средние у данных с текущей меткой по всем координатам
    means = data_train_means.loc[current_label].copy()
    #сдвигаем на средние по каждой координате
    biased = data_unlabeled - means
    #проходим по всем парам координат и переводим в полярные координаты
    polar = pd.DataFrame()
    for j in range(int(biased.shape[1] / 2)):
      t = biased.iloc[:, [2*j, 2*j+1]].copy().apply(convert_to_polar, axis='columns')
      polar = pd.concat([polar, t], axis = 1)
    polar_data.append(polar)
  return polar_data
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# построение кусочно постоянной функции
def func_def(target, data_train, polar_data, D):
    list_of_labels = data_train[target].unique()
    list_of_labels.sort()
    #число делений отрезка 0; 2pi
    Dlist = [2*np.pi * j /D for j in range(D+1)]
    dim = polar_data[0].shape[1]
    #для каждой метки строим функцию и раскладываем ее в ряд фурье
    labeled_func = []
    for i in range(len(list_of_labels)):
        piece_wise_func = []
        for p in range(int(dim/2)):
            func_dim = []
            #максимумы на отрезках
            for j in range(D):     
                func_dim.append(polar_data[i].iloc[:, [2*p, 2*p+1]].loc[(polar_data[i].iloc[:, 2*p] >= Dlist[j]) & (polar_data[i].iloc[:, 2*p] <= Dlist[j+1])].max())
            
            #вычисление функции на отрезках, где нет ни одной точки
            temp = func_dim.copy()
            #присваиваем четным координатам соответствующие значения на отрезке от 0 до 2 пи
            for j in range(D):
                temp[j].iloc[0] = 2*np.pi * j /D
            #причем новые "восстановленные данные" НЕ участвуют в восстановлении последующих
            for j in range(D):
                if np.isnan(func_dim[j].iloc[1]):
                    k=1
                    while k<=D:
                        left = (j-k+D)%D
                        right = (k+j)%D
                        if not np.isnan(func_dim[left].iloc[1]) and not np.isnan(func_dim[right].iloc[1]):
                            ar_mean = (func_dim[left].iloc[1] + func_dim[right].iloc[1]) / 2
                            temp[j].iloc[1] = ar_mean
                            break
                        elif not np.isnan(func_dim[left].iloc[1]):
                            ar_mean = func_dim[left].iloc[1]
                            temp[j].iloc[1] = ar_mean
                            break
                        elif not np.isnan(func_dim[right].iloc[1]):
                            ar_mean = func_dim[right].iloc[1]
                            temp[j].iloc[1] = ar_mean
                            break
                        else:
                            k+=1
            piece_wise_func.append(temp)
        labeled_func.append(piece_wise_func)
    return labeled_func
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# вычисление коэффициентов ряда фурье
def fourier_coef_calc(target, data_train, labeled_piece_func, D, M):
    list_of_labels = data_train[target].unique()
    list_of_labels.sort()
    dim = int((data_train.shape[1] - 1) / 2)
    row_labels = [0]
    for idx in range (1, M):
        row_labels.append(idx)
        row_labels.append(-idx)
    #для каждой метки
    labeled_coef = []
    for i in range(len(list_of_labels)):
        #для каждой пары координат от 0 до 392
        dim_coef = []
        for p in range(int(dim)):
            coef = pd.DataFrame(complex(0), index = row_labels, columns = ['value'])
            for idx in row_labels:
                sum = complex(0)
                #если коэффициент с индексом 0
                if idx == 0:
                    for j in range(D):
                        g = labeled_piece_func[i][p][j].iloc[1]
                        sum += g
                    sum /= D
                #если коэффициент с ненулевым индексом
                else:
                    for j in range(D):
                        g = labeled_piece_func[i][p][j].iloc[1]
                        a = 2*np.pi * j /D
                        b = 2*np.pi * (j + 1) /D
                        left_number = complex( np.cos(-b* idx), np.sin(-b* idx))
                        right_number = complex( np.cos(-a* idx), np.sin(-a* idx))
                        denominator = complex(imag = - idx)
                        sum += g* (left_number - right_number) / denominator
                    sum /= 2*np.pi
                coef.loc[idx] = sum
            dim_coef.append(coef)
        labeled_coef.append(dim_coef)
    return labeled_coef
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# классификация
def get_label(target, data_train, data_train_means, fourier_coef, image):
    list_of_labels = data_train[target].unique()
    list_of_labels.sort()
    dim = int((data_train.shape[1] - 1) / 2)
    votes = [0 for i in range (len(list_of_labels))]
    for i in range(len(list_of_labels)):
        current_label = list_of_labels[i]
        temp = image.drop(columns=[target])
        #сдвигаем на среднее
        means = data_train_means.loc[current_label].copy()
        biased = temp - means
        polar = pd.DataFrame()
        for j in range(int(biased.shape[1] / 2)):
          t = biased.iloc[:, [2*j, 2*j+1]].copy().apply(convert_to_polar, axis='columns')
          polar = pd.concat([polar, t], axis = 1)
        for j in range(int(polar.shape[1] / 2)):
            vals = polar.iloc[0, 2*j:2*j+2].copy()
            phi = vals.iloc[0]
            r = vals.iloc[1]
            coefs = fourier_coef[i][j].copy()
            fourier_value = complex(0)
            indexes = coefs.index
            for idx in indexes:
                exp_num = complex(np.cos(idx * phi), np.sin(idx*phi))
                fourier_value += exp_num * coefs.loc[idx].iloc[0]
            real_v = fourier_value.real
            if r <= real_v:
                votes[i]+=1
    return max(range(len(votes)), key = lambda x: votes[x])
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# вывод изображения
def img_plot(img):
  plt.figure(figsize=(7,7))

  grid_data = img.values.copy().reshape(28, 28)
  plt.imshow(grid_data,interpolation= "none" , cmap="gray")
  plt.show()
# # # # # # # # # # # # # # # # # # # # # # # #