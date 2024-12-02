import numpy as np
import itertools


class LinearCode:

    def __init__(self, mat):
        self.mat = mat
        self.G = self.RREF()
        self.G = self.delete_null_rows()
        self.X = self.delete_lead_columns(self.lead())
        self.H = self.form_H(self.X, self.lead())

    def __set(self, m):
        self.mat = m


    # Приводит матрицу к ступенчатому виду по строкам (Row Echelon Form)
    def REF(self, n):
        """
            Приводит матрицу к ступенчатому виду по строкам (Row Echelon Form).
        """
        i_st = 0
        j = 0
        for i in range(n.shape[1]):
            j = i
            if 1 in n[:, j]:
                i_st = np.where(n[:, j] == 1)[0][0]
                break
        temp = np.copy(n[0, :])
        n[0, :] = n[i_st, :]
        n[i_st, :] = temp
        for i in np.where(n[:, j])[0][1:]:
            n[i, :] += n[0, :]
            n[i, :] = n[i, :] % 2
        if n.shape[0] == 1:
            return
        t = n[1:, :]
        self.REF(t)
        return n


    def RREF(self):
        """
        Приводит матрицу к приведенному ступенчатому виду (Reduced Row Echelon Form)
        """
        self.mat = self.REF(self.mat)
        for i in range(1, self.mat.shape[0]):
            index = np.where(self.mat[i, :] == 1)[0]
            if (index.shape[0] == 0):
                return self.mat
            else:
                index = index[0]
            for j in range(0, i):
                if self.mat[j, index] == 1:
                    self.mat[j, :] += self.mat[i, :]
                    self.mat[j, :] = self.mat[j, :] % 2
        return self.mat

    def delete_null_rows(self):
        """
        Удаляет полностью нулевые строки из матрицы.
        """
        for i in range(0, self.mat.shape[0]):
            if np.where(self.mat[i, :] == 1)[0].shape[0] == 0:
                self.mat = np.delete(self.mat, i, axis=0)
        return self.mat

    def lead(self):
        """
        Находит индексы ведущих столбцов (столбцов с ведущими единицами).
        """
        res = np.array([], dtype=int)
#        matrix = self.delete_null_rows()
        for i in range(0, self.mat.shape[0]):
            index = np.where(self.mat[i, :] == 1)[0][0]
            res = np.append(res, index)
        return res

    def delete_lead_columns(self, lead):
        """
        Удаляет ведущие столбцы из матрицы.
        """
        matrix = np.copy(self.mat)
        for i in range(lead.shape[0]):
            matrix = np.delete(matrix, lead[i] - i, axis=1)
        return matrix

    def form_H(self, temp, lead):
        """
        Формирует проверочную матрицу H на основе порождающей матрицы.
        :param temp:
        :param lead:
        :return:
        """
        id_matrix = np.eye(temp.shape[1])
        H = np.zeros((temp.shape[0] + temp.shape[1], temp.shape[1]), dtype=int)
        i_x = 0
        i_id = 0
        for i in range(H.shape[0]):
            if i in lead:
                H[i, :] = temp[i_x, :]
                i_x += 1
            else:
                H[i, :] = id_matrix[i_id, :]
                i_id += 1
        return H

    def generate_code_words_1(self):
        """
        Генерирует все кодовые слова путем сложения линейных комбинаций строк порождающей матрицы.
        :return:
        """
        res = set()
        for i in range(1, self.mat.shape[0] + 1):
            combinations = list(itertools.combinations(range(self.mat.shape[0]), i))
            for comb in combinations:
                word = np.zeros(self.mat.shape[1], dtype=int)
                for j in comb:
                    word += self.mat[j, :]
                word %= 2
                res.add(tuple(word.tolist()))
        return res

    def generate_code_words2(self):
        """
        Генерирует все кодовые слова путем умножения всех возможных двоичных слов на порождающую матрицу.
        :return:
        """
        res = []
        combinations = []
        a = self.mat.shape[1]
        stuff = [1, 2, 3, 4, 5]
        for i in range(1, self.mat.shape[1]):
            for subset in itertools.combinations(stuff, i):
                combinations.append(subset)

        for i in range(len(combinations)):
            word = np.zeros(self.mat.shape[0], dtype=int)
            for comb in combinations[i]:
                word[comb - 1] = 1
            res.append(word)
        for i in range(len(res)):
            res[i] = np.matmul(res[i], self.mat)
            res[i] %= 2
        return res

    def lines(self):
        """
        Считает количество ненулевых строк в текущей матрице.
        :return:
        """
        count = 0
        for row in self.mat:
            if sum(row) > 0:
                count += 1
        return count

    def columns(self):
        """
        Возвращает количество столбцов в текущей матрице.
        :return:
        """
        return len(self.mat[0])


# %%
s = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                  [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
o = LinearCode(s)

m = o.RREF()
m = o.delete_null_rows()

X = o.delete_lead_columns(o.lead())
print("print X ")
print(X)
H = o.form_H(X, o.lead())
print("print H ")
print(H)
print()
o.generate_code_words_1()
res = o.generate_code_words2()

for i in range(len(res)):
    res[i] = np.matmul(res[i], H)
    res[i] %= 2
print(res)
print("Columns", o.columns())
print("Lines ", o.lines())
print()
words = o.generate_code_words_1()

print(words)
print()
word = np.array([1, 1, 1, 1, 1])

after_coding = np.matmul(np.transpose(word), m)

after_decoding = np.matmul(after_coding, H) % 2

print(after_decoding)
print()
after_coding = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]

after_decoding = np.matmul(after_coding, H) % 2

print(after_decoding)
print()
after_coding = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

after_decoding = np.matmul(after_coding, H) % 2

print(after_decoding)
print()
print(after_coding)
print()
after_coding = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]) + np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

after_decoding = np.matmul(after_coding, H) % 2

print(after_decoding)
print()
print(H)
print()
print(m)
print()
word = np.array([1, 1, 1, 1, 1])
after_coding = np.matmul(np.transpose(word), o.G) % 2

print(after_coding)
print()
after_coding += np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

after_decoding = np.matmul(after_coding, o.H) % 2

print(after_decoding)
print()
e = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
np.matmul(e, o.H) % 2

