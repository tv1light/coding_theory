import numpy as np
import random
from itertools import combinations

# Функция для формирования проверочной матрицы Хэмминга
def generate_hamming_parity_check_matrix(r):
    n = 2 ** r - 1
    H = np.zeros((r, n), dtype=int)
    for i in range(1, n + 1):
        binary_repr = list(map(int, bin(i)[2:].zfill(r)))
        H[:, i - 1] = binary_repr[::-1]
    return H

# Функция для формирования порождающей матрицы Хэмминга
def generate_hamming_generator_matrix(r):
    n = 2 ** r - 1
    k = n - r
    H = generate_hamming_parity_check_matrix(r)
    P = H.T[:k, :]
    G = np.hstack((np.eye(k, dtype=int), P))
    return G

# Функция для формирования проверочной матрицы расширенного кода Хэмминга
def generate_extended_hamming_parity_check_matrix(r):
    n = 2 ** r
    H = np.zeros((r + 1, n), dtype=int)
    for i in range(1, n + 1):
        binary_repr = list(map(int, bin(i)[2:].zfill(r+1)[-r:]))
        H[:-1, i - 1] = binary_repr[::-1]
    H[-1, :] = np.sum(H[:-1, :], axis=0) % 2
    return H

# Функция для формирования порождающей матрицы расширенного кода Хэмминга
def generate_extended_hamming_generator_matrix(r):
    n = 2 ** r
    k = n - r - 1
    H = generate_extended_hamming_parity_check_matrix(r)
    P = H.T[:k, :-1]
    parity_column = np.sum(P, axis=1) % 2
    G = np.hstack((np.eye(k, dtype=int), P, parity_column.reshape(-1, 1)))
    return G

# Функция для генерации кодового слова с ошибками
def generate_codeword_with_errors(G, num_errors):
    k = G.shape[0]
    message = np.random.randint(0, 2, k)
    codeword = (message @ G) % 2
    error_positions = random.sample(range(len(codeword)), num_errors)
    for pos in error_positions:
        codeword[pos] ^= 1
    return codeword

# Функция для вычисления синдрома
def compute_syndrome(codeword, H):
    return (H @ codeword) % 2

# Функция для исправления ошибок в кодовом слове
def correct_errors(codeword, H, max_errors):
    n = len(codeword)
    syndrome = compute_syndrome(codeword, H)
    syndrome_tuple = tuple(syndrome)
    error_pattern = None

    # Перебор всех возможных комбинаций ошибок до max_errors
    for num_errors in range(1, max_errors + 1):
        for error_positions in combinations(range(n), num_errors):
            error_vector = np.zeros(n, dtype=int)
            error_vector[list(error_positions)] = 1
            test_syndrome = compute_syndrome(error_vector, H)
            if np.array_equal(test_syndrome, syndrome):
                error_pattern = error_vector
                break
        if error_pattern is not None:
            break

    if error_pattern is not None:
        corrected_codeword = (codeword + error_pattern) % 2
        return corrected_codeword
    else:
        print("Не удалось исправить ошибки.")
        return codeword

# Функция для исследования кода Хэмминга
def hamming_code_investigation(r):
    print(f"\nИсследование кода Хэмминга для r = {r}")
    G = generate_hamming_generator_matrix(r)
    H = generate_hamming_parity_check_matrix(r)
    n, k = G.shape[1], G.shape[0]
    print("Порождающая матрица G:")
    print(G)
    print("Проверочная матрица H:")
    print(H)

    for num_errors in range(1, 4):
        print(f"\nКодовое слово с {num_errors} ошибкой(ами):")
        codeword_with_errors = generate_codeword_with_errors(G, num_errors)
        print("Кодовое слово с ошибками:")
        print(codeword_with_errors)
        corrected_codeword = correct_errors(codeword_with_errors, H, num_errors)
        print("Исправленное кодовое слово:")
        print(corrected_codeword)
        syndrome_after_correction = compute_syndrome(corrected_codeword, H)
        print("Синдром после исправления ошибок:")
        print(syndrome_after_correction)

# Функция для исследования расширенного кода Хэмминга
def extended_hamming_code_investigation(r):
    print(f"\nИсследование расширенного кода Хэмминга для r = {r}")
    G = generate_extended_hamming_generator_matrix(r)
    H = generate_extended_hamming_parity_check_matrix(r)
    n, k = G.shape[1], G.shape[0]
    print("Порождающая матрица G:")
    print(G)
    print("Проверочная матрица H:")
    print(H)

    for num_errors in range(1, 5):
        print(f"\nКодовое слово с {num_errors} ошибкой(ами):")
        codeword_with_errors = generate_codeword_with_errors(G, num_errors)
        print("Кодовое слово с ошибками:")
        print(codeword_with_errors)
        corrected_codeword = correct_errors(codeword_with_errors, H, num_errors)
        print("Исправленное кодовое слово:")
        print(corrected_codeword)
        syndrome_after_correction = compute_syndrome(corrected_codeword, H)
        print("Синдром после исправления ошибок:")
        print(syndrome_after_correction)

if __name__ == '__main__':
    # Исследование кодов Хэмминга для r = 2, 3, 4
    for r in [2, 3, 4]:
        hamming_code_investigation(r)

    # Исследование расширенных кодов Хэмминга для r = 2, 3, 4
    for r in [2, 3, 4]:
        extended_hamming_code_investigation(r)

