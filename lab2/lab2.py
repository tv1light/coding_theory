import copy
import random


def identity_matrix(k):
    """
    Создает единичную матрицу размером k x k.
    """
    return [[1 if i == j else 0 for j in range(k)] for i in range(k)]


def vector_sum(vector):
    """
    Вычисляет сумму элементов вектора.
    """
    return sum(vector)


def multiply_vector_matrix(vector, matrix):
    """
    Перемножает вектор с матрицей по модулю 2.
    """
    result = []
    for col in zip(*matrix):
        product = sum(v * m for v, m in zip(vector, col)) % 2
        result.append(product)
    return result


def add_vectors(v1, v2):
    """
    Складывает два вектора по модулю 2.
    """
    return [(a + b) % 2 for a, b in zip(v1, v2)]


def is_vector_in_matrix(vector, matrix):
    """
    Проверяет, находится ли вектор в матрице.
    """
    return any(vector == row for row in matrix)


def get_vector_index(vector, matrix):
    """
    Возвращает индекс вектора в матрице, если он существует, иначе -1.
    """
    try:
        return matrix.index(vector)
    except ValueError:
        return -1


def concatenate_horizontal(matrix1, matrix2):
    """
    Объединяет две матрицы по горизонтали.
    """
    return [row1 + row2 for row1, row2 in zip(matrix1, matrix2)]


def concatenate_vertical(matrix1, matrix2):
    """
    Объединяет две матрицы по вертикали.
    """
    return copy.deepcopy(matrix1) + copy.deepcopy(matrix2)


def generate_X(k, n):
    """
    Генерирует матрицу X, удовлетворяющую условиям минимального расстояния d=5.
    """
    while True:
        X = [[random.randint(0, 1) for _ in range(n)] for _ in range(k)]

        # Условие а: не менее 4 единиц в каждой строке
        if not all(sum(row) >= 4 for row in X):
            continue

        # Условие б: сумма любых двух строк содержит не менее 3 единиц
        if any(sum(add_vectors(X[i], X[j])) < 3 for i in range(k) for j in range(i + 1, k)):
            continue

        # Условие в: сумма любых трех строк содержит не менее 2 единиц
        if any(sum(add_vectors(add_vectors(X[i], X[j]), X[m])) < 2
               for i in range(k) for j in range(i + 1, k) for m in range(j + 1, k)):
            continue

        # Условие г: сумма любых четырех строк содержит не менее 1 единицу
        if any(sum(add_vectors(add_vectors(add_vectors(X[i], X[j]), X[m]), X[l])) < 1
               for i in range(k) for j in range(i + 1, k)
               for m in range(j + 1, k) for l in range(m + 1, k)):
            continue

        return X


def generate_closure(generator_matrix):
    """
    Генерирует замыкание подпространства, порожденного порождающей матрицей.
    """
    closure = []
    for col in zip(*generator_matrix):
        vector = list(col)
        if vector not in closure:
            closure.append(vector)

    added = True
    while added:
        added = False
        new_vectors = []
        for i in range(len(closure)):
            for j in range(i + 1, len(closure)):
                summed = add_vectors(closure[i], closure[j])
                if summed not in closure and summed not in new_vectors:
                    new_vectors.append(summed)
                    added = True
        closure.extend(new_vectors)

    return closure


def generate_error_vector(n, num_errors):
    """
    Генерирует вектор ошибок с заданным количеством ошибок.
    """
    error = [0] * n
    error_indices = random.sample(range(n), num_errors)
    for idx in error_indices:
        error[idx] = 1
    return error


def get_codewords(closure, generator_matrix):
    """
    Генерирует кодовые слова из замыкания и порождающей матрицы.
    """
    return [multiply_vector_matrix(vec, generator_matrix) for vec in closure]


def correct_single_error(parity_check_matrix, syndrome, word):
    """
    Исправляет одноранговую ошибку в кодовом слове.
    """
    index = get_vector_index(syndrome, parity_check_matrix)
    if index != -1:
        word[index] = (word[index] + 1) % 2
    else:
        print("Синдром не найден в проверочной матрице H.")
    return word


def correct_double_error(parity_check_matrix, syndrome, word):
    """
    Пытается исправить двукратные ошибки в кодовом слове.
    """
    for i in range(len(parity_check_matrix)):
        if syndrome == parity_check_matrix[i]:
            word[i] = (word[i] + 1) % 2
            return word
        for j in range(i + 1, len(parity_check_matrix)):
            combined_syndrome = add_vectors(parity_check_matrix[i], parity_check_matrix[j])
            if syndrome == combined_syndrome:
                word[i] = (word[i] + 1) % 2
                word[j] = (word[j] + 1) % 2
                return word
    print("Синдром не найден для исправления двух ошибок.")
    return word


def first_part():
    """
    Выполняет задания Часть 1 лабораторной работы.
    """
    K = 4
    N = 7
    print("\n----------------------------Часть 1-----------------------\n")

    # Задание 2.1: Порождающая матрица G
    print("Матрица X:")
    X = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    for row in X:
        print(row)

    G = concatenate_horizontal(identity_matrix(K), X)
    print("\nПорождающая матрица G:")
    for row in G:
        print(row)

    # Задание 2.2: Проверочная матрица H
    H = concatenate_vertical(X, identity_matrix(N - K))
    print("\nПроверочная матрица H:")
    for row in H:
        print(row)

    # Замыкание подпространства U
    closure = generate_closure(G)
    print("\nЗамыкание подпространства U:")
    print(closure[0])

    # Генерация первого кодового слова
    codeword = multiply_vector_matrix(closure[0], G)
    print("\nКодовое слово:")
    print(codeword)

    # Добавление одной ошибки
    error1 = generate_error_vector(N, 1)
    print("\nВектор ошибки (1 ошибка):")
    print(error1)

    received_word1 = add_vectors(codeword, error1)
    print("\nКодовое слово с одной ошибкой:")
    print(received_word1)

    syndrome1 = multiply_vector_matrix(received_word1, H)
    print("\nСиндром для слова с одной ошибкой:")
    print(syndrome1)

    # Корректировка одной ошибки
    corrected_word1 = correct_single_error(H, syndrome1, received_word1)
    print("\nИсправленное кодовое слово (1 ошибка):")
    print(corrected_word1)

    # Проверка
    print("\nПроверка синдрома исправленного слова:")
    print(multiply_vector_matrix(corrected_word1, H))

    # Добавление двух ошибок
    error2 = generate_error_vector(N, 2)
    print("\nВектор ошибки (2 ошибки):")
    print(error2)

    received_word2 = add_vectors(codeword, error2)
    print("\nКодовое слово с двумя ошибками:")
    print(received_word2)

    syndrome2 = multiply_vector_matrix(received_word2, H)
    print("\nСиндром для слова с двумя ошибками:")
    print(syndrome2)

    # Корректировка двух ошибок
    corrected_word2 = correct_double_error(H, syndrome2, received_word2)
    print("\nИсправленное кодовое слово (2 ошибки):")
    print(corrected_word2)

    # Проверка
    print("\nПроверка синдрома исправленного слова:")
    print(multiply_vector_matrix(corrected_word2, H))


def second_part():
    """
    Выполняет задания Часть 2 лабораторной работы.
    """
    N = 11
    K = 4
    print("\n----------------------------Часть 2-----------------------\n")

    # Задание 2.6: Генерация матрицы X для d=5
    print("Матрица X (генерируется для d=5):")
    X_second = generate_X(K, N - K)
    for row in X_second:
        print(row)

    # Порождающая матрица G
    G = concatenate_horizontal(identity_matrix(K), X_second)
    print("\nПорождающая матрица G:")
    for row in G:
        print(row)

    # Проверочная матрица H
    H = concatenate_vertical(X_second, identity_matrix(N - K))
    print("\nПроверочная матрица H:")
    for row in H:
        print(row)

    # Замыкание подпространства U
    closure = generate_closure(G)
    print("\nЗамыкание подпространства U:")
    print(closure[0])

    # Генерация кодового слова
    codeword = multiply_vector_matrix(closure[0], G)
    print("\nКодовое слово:")
    print(codeword)

    # Добавление одной ошибки
    error1 = generate_error_vector(N, 1)
    print("\nВектор ошибки (1 ошибка):")
    print(error1)

    received_word1 = add_vectors(codeword, error1)
    print("\nКодовое слово с одной ошибкой:")
    print(received_word1)

    syndrome1 = multiply_vector_matrix(received_word1, H)
    print("\nСиндром для слова с одной ошибкой:")
    print(syndrome1)

    # Корректировка одной ошибки
    corrected_word1 = correct_single_error(H, syndrome1, received_word1)
    print("\nИсправленное кодовое слово (1 ошибка):")
    print(corrected_word1)

    # Проверка
    print("\nПроверка синдрома исправленного слова:")
    print(multiply_vector_matrix(corrected_word1, H))

    # Добавление двух ошибок
    error2 = generate_error_vector(N, 2)
    print("\nВектор ошибки (2 ошибки):")
    print(error2)

    received_word2 = add_vectors(codeword, error2)
    print("\nКодовое слово с двумя ошибками:")
    print(received_word2)

    syndrome2 = multiply_vector_matrix(received_word2, H)
    print("\nСиндром для слова с двумя ошибками:")
    print(syndrome2)

    # Корректировка двух ошибок
    corrected_word2 = correct_double_error(H, syndrome2, received_word2)
    print("\nИсправленное кодовое слово (2 ошибки):")
    print(corrected_word2)

    # Проверка
    print("\nПроверка синдрома исправленного слова:")
    print(multiply_vector_matrix(corrected_word2, H))

    # Добавление трёх ошибок
    error3 = generate_error_vector(N, 3)
    print("\nВектор ошибки (3 ошибки):")
    print(error3)

    received_word3 = add_vectors(codeword, error3)
    print("\nКодовое слово с тремя ошибками:")
    print(received_word3)

    syndrome3 = multiply_vector_matrix(received_word3, H)
    print("\nСиндром для слова с тремя ошибками:")
    print(syndrome3)

    # Попытка исправления трёх ошибок (будет ошибкой или не исправит)
    corrected_word3 = correct_double_error(H, syndrome3, received_word3)
    print("\nИсправленное кодовое слово (3 ошибки):")
    print(corrected_word3)

    # Проверка
    print("\nПроверка синдрома исправленного слова:")
    print(multiply_vector_matrix(corrected_word3, H))


if __name__ == "__main__":
    first_part()
    second_part()
