import numpy as np
from itertools import combinations, product
import random

def get_basis(m):
    """Возвращает все двоичные векторы длины m."""
    return np.array(list(product([0, 1], repeat=m)), dtype=int)

def get_monomial_codeword(I, m):
    """
    Получает кодовое слово, соответствующее моному, заданному I.

    Параметры:
    - I: кортеж индексов, представляющих переменные в мономе
    - m: общее количество переменных

    Возвращает:
    - numpy массив, представляющий кодовое слово
    """
    x = get_basis(m)
    if len(I) == 0:
        return np.ones(2 ** m, dtype=int)
    else:
        return np.prod(x[:, I], axis=1)  # Произведение по переменным в I

def generate_reed_muller_G(r, m):
    """
    Генерирует порождающую матрицу G кода Рида-Маллера (r, m).

    Параметры:
    - r: порядок кода Рида-Маллера
    - m: количество переменных

    Возвращает:
    - numpy массив, представляющий порождающую матрицу G
    """
    monomials = []
    variables = range(m)
    for deg in range(r + 1):
        for I in combinations(variables, deg):
            codeword = get_monomial_codeword(I, m)
            monomials.append(codeword)
    G = np.array(monomials, dtype=int)
    return G

def generate_codeword(u, G):
    """
    Генерирует кодовое слово, соответствующее сообщению u с использованием порождающей матрицы G.

    Параметры:
    - u: вектор сообщения
    - G: порождающая матрица

    Возвращает:
    - numpy массив, представляющий кодовое слово
    """
    codeword = np.mod(u @ G, 2)
    return codeword

def introduce_errors(codeword, error_count):
    """
    Вносит ошибки в случайных позициях кодового слова.

    Параметры:
    - codeword: исходное кодовое слово
    - error_count: количество ошибок для внесения

    Возвращает:
    - кортеж, содержащий полученное слово с ошибками и позиции ошибок
    """
    n = len(codeword)
    error_positions = random.sample(range(n), error_count)
    received = codeword.copy()
    received[error_positions] ^= 1  # Инвертировать биты в позициях ошибок
    return received, error_positions

def majority_decode(received, r, m):
    """
    Алгоритм мажоритарного декодирования для кода Рида-Маллера (r, m).

    Параметры:
    - received: принятое кодовое слово с возможными ошибками
    - r: порядок кода Рида-Маллера
    - m: количество переменных

    Возвращает:
    - numpy массив, представляющий декодированное сообщение
    """
    decoded_message = []
    # Начинаем с мономов наивысшей степени
    for deg in reversed(range(r + 1)):
        variables = range(m)
        monomials = list(combinations(variables, deg))
        for I in monomials:
            monomial_cw = get_monomial_codeword(I, m)
            # Вычисляем скалярное произведение над GF(2)
            inner_product = np.dot(received, monomial_cw) % 2
            # Мажоритарная логика: проверяем, превышает ли вес порог
            threshold = 2 ** (m - deg - 1)
            weight = np.sum(received[monomial_cw == 1])
            if weight >= threshold:
                decoded_bit = 1
                # Вычитаем мономиальное кодовое слово из принятого слова
                received = (received + monomial_cw) % 2
            else:
                decoded_bit = 0
            decoded_message.insert(0, decoded_bit)  # Добавляем в начало для сохранения правильного порядка
    return np.array(decoded_message, dtype=int)

def main():
    m, r = 4, 2
    G = generate_reed_muller_G(r, m)
    print("Порождающая матрица G({},{})\n".format(r, m), G)
    print("_____________________________\n")

    # Генерируем случайное сообщение
    message_length = G.shape[0]
    u = np.random.randint(0, 2, size=message_length)
    print("Исходное сообщение u:\n", u)
    codeword = generate_codeword(u, G)
    print("\nКодовое слово:\n", codeword)

    # Эксперимент с одной ошибкой
    received, error_positions = introduce_errors(codeword, 1)
    print("\nПринятое слово с одной ошибкой в позициях {}:\n".format(error_positions), received)

    decoded_u = majority_decode(received.copy(), r, m)
    print("\nДекодированное сообщение u_hat:\n", decoded_u)
    recovered_codeword = generate_codeword(decoded_u, G)
    print("\nВосстановленное кодовое слово:\n", recovered_codeword)
    print("\nУспешная коррекция ошибок:", np.array_equal(recovered_codeword, codeword))

    # Эксперимент с двумя ошибками
    print("_____________________________\n")
    received_double, error_positions_double = introduce_errors(codeword, 2)
    print("Принятое слово с двумя ошибками в позициях {}:\n".format(error_positions_double), received_double)

    decoded_u_double = majority_decode(received_double.copy(), r, m)
    print("\nДекодированное сообщение u_hat:\n", decoded_u_double)
    recovered_codeword_double = generate_codeword(decoded_u_double, G)
    print("\nВосстановленное кодовое слово:\n", recovered_codeword_double)
    print("\nУспешная коррекция ошибок:", np.array_equal(recovered_codeword_double, codeword))

if __name__ == '__main__':
    main()
