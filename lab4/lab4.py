import random
import numpy as np

import numpy as np
import random
from itertools import combinations

# Расширенный код Голея (24,12,8)
class ExtendedGolayCode:
    """
    Класс для работы с расширенным кодом Голея (24,12,8)
    """

    def __init__(self):
        self.n = 24  # длина кода
        self.k = 12  # длина информационного слова
        self.G = self._construct_generator_matrix()
        self.H = self._construct_parity_check_matrix()

    def _construct_generator_matrix(self):
        """
        Формирование порождающей матрицы G
        """
        b_matrix = np.array([
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ])
        I_k = np.eye(self.k, dtype=int)
        G = np.hstack((I_k, b_matrix))
        return G

    def _construct_parity_check_matrix(self):
        """
        Формирование проверочной матрицы H
        """
        b_matrix = self.G[:, self.k:]
        H = np.vstack((b_matrix.T, np.eye(self.n - self.k, dtype=int))).T
        return H

    def encode(self, message):
        """
        Кодирование сообщения
        """
        codeword = np.dot(message, self.G) % 2
        return codeword

    def introduce_errors(self, codeword, num_errors):
        """
        Внесение ошибок в кодовое слово
        """
        error_positions = random.sample(range(self.n), num_errors)
        codeword_with_errors = codeword.copy()
        codeword_with_errors[error_positions] ^= 1
        return codeword_with_errors, error_positions

    def calculate_syndrome(self, codeword):
        """
        Вычисление синдрома
        """
        syndrome = np.dot(codeword, self.H.T) % 2
        return syndrome

    def correct_errors(self, codeword_with_errors, max_errors=3):
        """
        Исправление ошибок в кодовом слове (до max_errors ошибок)
        """
        syndrome = self.calculate_syndrome(codeword_with_errors)
        if not syndrome.any():
            return codeword_with_errors  # Нет ошибок

        H_T = self.H.T
        n = self.n

        # Поиск ошибок
        for num_errors in range(1, max_errors + 1):
            for error_positions in combinations(range(n), num_errors):
                error_pattern = np.zeros(n, dtype=int)
                error_pattern[list(error_positions)] = 1
                syndrome_candidate = np.dot(error_pattern, self.H.T) % 2
                if np.array_equal(syndrome_candidate, syndrome):
                    corrected_codeword = codeword_with_errors.copy()
                    corrected_codeword[list(error_positions)] ^= 1
                    return corrected_codeword
        print("Не удалось исправить ошибки")
        return codeword_with_errors  # Возврат без исправления

    def generate_random_message(self):
        """
        Генерация случайного информационного сообщения
        """
        return np.random.randint(0, 2, self.k)


# Код Рида-Маллера RM(r, m)
class ReedMullerCode:
    """
    Класс для работы с кодами Рида-Маллера RM(r, m)
    """

    def __init__(self, r, m):
        self.r = r  # порядок кода
        self.m = m  # количество переменных
        self.n = 2 ** m  # длина кода



def test_extended_golay_code():
    print("Часть 1: Расширенный код Голея (24,12,8)\n")
    code = ExtendedGolayCode()
    print("Порождающая матрица G:\n", code.G)
    print("\nПроверочная матрица H:\n", code.H)

    for num_errors in [1, 2, 3, 4]:
        print(f"\n--- Тестирование с {num_errors} ошибками ---")
        message = code.generate_random_message()
        print("Исходное сообщение:", message)
        codeword = code.encode(message)
        print("Кодовое слово:", codeword)
        codeword_with_errors, error_positions = code.introduce_errors(codeword, num_errors)
        print(f"Кодовое слово с ошибками (позиции ошибок {error_positions}):", codeword_with_errors)
        corrected_codeword = code.correct_errors(codeword_with_errors, max_errors=3)
        print("Исправленное кодовое слово:", corrected_codeword)
        if np.array_equal(corrected_codeword, codeword):
            print("Ошибки успешно исправлены.")
        else:
            print("Не удалось исправить ошибки полностью.")
        syndrome_after = code.calculate_syndrome(corrected_codeword)
        print("Синдром после исправления:", syndrome_after)

# Код Рида-Маллера RM(r, m)
class ReedMullerCode:
    """
    Класс для работы с кодами Рида-Маллера RM(r, m)
    """

    def __init__(self, r, m):
        self.r = r  # порядок кода
        self.m = m  # количество переменных
        self.n = 2 ** m  # длина кода
        self.G = self.generate_generator_matrix()

    def generate_generator_matrix(self):
        """
        Генерация порождающей матрицы для RM(r, m)
        """
        if self.r == 0:
            return np.ones((1, self.n), dtype=int)
        elif self.r == self.m:
            return np.eye(self.n, dtype=int)
        else:
            G_rm_minus_1 = ReedMullerCode(self.r, self.m - 1).G
            G_r_minus_1 = ReedMullerCode(self.r - 1, self.m - 1).G
            upper = np.hstack((G_rm_minus_1, G_rm_minus_1))
            lower = np.hstack((np.zeros_like(G_r_minus_1), G_r_minus_1))
            G = np.vstack((upper, lower))
            return G

    def encode(self, message):
        """
        Кодирование сообщения
        """
        codeword = np.dot(message, self.G) % 2
        return codeword

    def introduce_errors(self, codeword, num_errors):
        """
        Внесение ошибок в кодовое слово
        """
        error_positions = random.sample(range(self.n), num_errors)
        codeword_with_errors = codeword.copy()
        codeword_with_errors[error_positions] ^= 1
        return codeword_with_errors, error_positions

    def decode(self, received_word):
        """
        Декодирование кодового слова (методом большинства)
        """
        # Преобразование в значения -1 и 1
        received_word = np.where(received_word == 0, -1, 1)
        # Инициализация
        s = received_word.copy()
        for i in range(self.r):
            s = self._majority_logic_decoding(s)
        # Преобразование обратно в 0 и 1
        decoded_codeword = np.where(s == -1, 0, 1)
        return decoded_codeword

    def _majority_logic_decoding(self, s):
        """
        Одноуровневое декодирование на основе алгоритма большинства
        """
        n = s.size
        s_new = np.zeros(n // 2, dtype=int)
        for i in range(0, n, 2):
            s_new[i // 2] = np.sign(s[i] + s[i + 1])
        return s_new

    def generate_random_message(self):
        """
        Генерация случайного информационного сообщения
        """
        message_length = self.G.shape[0]
        return np.random.randint(0, 2, message_length)

def test_reed_muller_code(r, m, error_cases):
    print(f"\nЧасть 2: Код Рида-Маллера RM({r},{m})\n")
    code = ReedMullerCode(r, m)
    print("Порождающая матрица G:\n", code.G)

    for num_errors in error_cases:
        print(f"\n--- Тестирование с {num_errors} ошибками ---")
        message = code.generate_random_message()
        print("Исходное сообщение:", message)
        codeword = code.encode(message)
        print("Кодовое слово:", codeword)
        codeword_with_errors, error_positions = code.introduce_errors(codeword, num_errors)
        print(f"Кодовое слово с ошибками (позиции ошибок {error_positions}):", codeword_with_errors)
        decoded_codeword = code.decode(codeword_with_errors)
        print("Декодированное кодовое слово:", decoded_codeword)
        if np.array_equal(decoded_codeword, codeword):
            print("Ошибки успешно исправлены.")
        else:
            print("Не удалось исправить ошибки полностью.")


if __name__ == "__main__":
    # Часть 1: Расширенный код Голея
    test_extended_golay_code()

    # Часть 2: Код Рида-Маллера RM(1,3)
    test_reed_muller_code(r=1, m=3, error_cases=[1, 2])

    # Часть 2: Код Рида-Маллера RM(1,4)    test_reed_muller_code(r=1, m=4, error_cases=[1, 2, 3, 4])