import numpy as np
import random


def generate_random_word(length):
    """Генерация случайного двоичного слова."""
    return np.random.randint(0, 2, length, dtype=int)


def encode_cyclic_code(u, g):
    """Кодирование слова циклическим кодом."""
    encoded = np.polymul(u, g) % 2
    return encoded.astype(int)


def introduce_random_errors(codeword, error_count):
    """Добавление случайных ошибок в закодированное слово."""
    positions = random.sample(range(len(codeword)), error_count)
    for pos in positions:
        codeword[pos] = (codeword[pos] + 1) % 2
    return codeword


def introduce_packet_error(codeword, packet_length):
    """Добавление пакетной ошибки в закодированное слово."""
    start_pos = random.randint(0, len(codeword) - packet_length)
    for i in range(packet_length):
        if start_pos + i < len(codeword):
            codeword[start_pos + i] = (codeword[start_pos + i] + random.randint(0, 1)) % 2
    return codeword


def decode_cyclic_code(w, g, t, is_packet):
    """Декодирование циклического кода."""
    n = len(w)
    syndrome = np.polydiv(w, g)[1] % 2
    if np.all(syndrome == 0):  # Если синдром равен нулю, ошибок нет.
        return np.polydiv(w, g)[0] % 2

    for i in range(n):
        error_pattern = np.zeros(n, dtype=int)
        error_pattern[-(i + 1)] = 1
        error_syndrome = np.polymul(syndrome, error_pattern) % 2
        residual = np.polydiv(error_syndrome, g)[1] % 2

        if is_packet:
            if len(residual) <= t:
                return correct_error(w, residual, g)
        elif np.sum(residual) <= t:
            return correct_error(w, residual, g)
    return None


def correct_error(w, error_syndrome, g):
    """Коррекция ошибок."""
    corrected = np.polyadd(w, error_syndrome) % 2
    return np.polydiv(corrected, g)[0] % 2


if __name__ == '__main__':
    # Код (7,4)
    g1 = np.array([1, 0, 1, 1], dtype=int)
    n1, k1, t1 = 7, 4, 1

    print("Код (7,4):")
    for errors in range(1, 4):
        u = generate_random_word(k1)
        print(f"Исходное слово: {u}")
        encoded = encode_cyclic_code(u, g1)
        print(f"Закодированное слово: {encoded}")
        noisy = introduce_random_errors(encoded.copy(), errors)
        print(f"Слово с {errors} ошибками: {noisy}")
        decoded = decode_cyclic_code(noisy, g1, t1, is_packet=False)
        print(f"Декодированное слово: {decoded}")
        print()

    # Код (15,9)
    g2 = np.array([1, 0, 0, 1, 1, 1, 1], dtype=int)
    n2, k2, t2 = 15, 9, 3

    print("Код (15,9):")
    for packet_length in range(1, 5):
        u = generate_random_word(k2)
        print(f"Исходное слово: {u}")
        encoded = encode_cyclic_code(u, g2)
        print(f"Закодированное слово: {encoded}")
        noisy = introduce_packet_error(encoded.copy(), packet_length)
        print(f"Слово с пакетом ошибок длины {packet_length}: {noisy}")
        decoded = decode_cyclic_code(noisy, g2, t2, is_packet=True)
        print(f"Декодированное слово: {decoded}")
        print()
