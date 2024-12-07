# Coding_theory  
# Лабораторная №1
**1.1. Реализовать функцию REF(), приводящую матрицу к
ступенчатому виду.**  
Ступенчатой матрицей, или матрицей ступенчатого вида по строкам, называется
матрица, такая что  
• все ненулевые строки (имеющие по крайней мере один ненулевой элемент)
располагаются над всеми чисто нулевыми строками  
• ведущий элемент (первый, считая слева направо, ненулевой элемент строки)
каждой ненулевой строки располагается строго правее ведущего элемента в
строке, расположенной выше данной.  
[[1 0 1 1 0 0 0 1 0 0 1]  
[0 0 0 1 1 1 0 1 0 1 0]  
[0 0 0 0 1 0 0 1 0 0 1]  
[0 0 0 0 0 0 1 0 0 1 0]  
[0 0 0 0 0 0 0 0 1 1 1]]  
Бирюзовым цветом указаны ведущие элементы, жёлтым – элементы под ведущими,
зелёным – над ведущими. Обратите внимание, что все элементы, отмеченные
жёлтым – равны нулю, но часть зелёных – нет.
Допустимо пользоваться numpy для базовых операций с матрицами. Значения
элементов матрицы должны быть булевскими либо целочисленными. В первом
случае сложение выполняется как исключающее ИЛИ (XOR). Во втором операции
над строками должны выполняться по модулю 2. Полностью нулевые строки можно
удалить.  
Обратите внимание, что в зависимости от разработанного вами алгоритма
конкретные значения строк и их порядок в полученной матрице может меняться.  

**1.2. Реализовать функцию RREF(), приводящую матрицу к  
приведённому ступенчатому виду.**  
Ступенчатая матрица называется приведенной, если матрица, составленная из
всех ее основных столбцов, является единичной матрицей (столбец матрицы
называется основным, если он содержит ведущий элемент какой-либо строки
матрицы).  
То есть, приведенная ступенчатая матрица не имеет нулевых строк, и все ведущие
элементы ее строк равны единице. При этом все элементы основных столбцов,
помимо ведущих элементов, являются нулями.  
[[1 0 1 0 0 1 0 1 0 1 0]  
[0 0 0 1 0 1 0 0 0 1 1]  
[0 0 0 0 1 0 0 1 0 0 1]  
[0 0 0 0 0 0 1 0 0 1 0]  
[0 0 0 0 0 0 0 0 1 1 1]]  
Бирюзовым цветом указаны ведущие элементы, жёлтым – элементы под ведущими,
зелёным – над ведущими. Обратите внимание, что все элементы, отмеченные
жёлтым и зелёным – равны нулю.  
См. уточнение к заданию 1.  

**1.3. Создать класс линейных кодов LinearCode**.  
Для инициализации линейного кода используется множество векторов
одной длины (можно представить в форме матрицы).  

**1.3.1 На основе входной матрицы сформировать порождающую  
матрицу в ступенчатом виде.**

**1.3.2 Задать n равное числу столбцов и k равное числу строк
полученной матрицы (без учёта полностью нулевых строк).**  

**1.3.3 Сформировать проверочную матрицу на основе порождающей.**  

# Лабораторная №2
Часть 1.  
**2.1. Сформировать порождающую матрицу линейного кода (7, 4, 3).**  
**2.2 Сформировать проверочную матрицу на основе порождающей.**  
**2.3 Сформировать таблицу синдромов для всех однократных  
ошибок.**  
**2.4. Сформировать кодовое слово длины n из слова длины k.** Внести  
однократную ошибку в сформированное слово. Вычислить синдром,  
исправить ошибку с использованием таблицы синдромов. Убедиться  
в правильности полученного слова.  
**2.5. Сформировать кодовое слово длины n из слова длины k.** Внести  
двукратную ошибку в сформированное слово. Вычислить синдром,  
исправить ошибку с использованием таблицы синдромов.  
Убедиться, что полученное слово отличается от отправленного.  

Часть 2.  
**2.6. Сформировать порождающую матрицу линейного кода (n, k, 5).**  
**2.7 Сформировать проверочную матрицу на основе порождающей.**  
**2.8 Сформировать таблицу синдромов для всех однократных и  
двукратных ошибок.**  
**2.9. Сформировать кодовое слово длины n из слова длины k.** Внести  
однократную ошибку в сформированное слово. Вычислить синдром,  
исправить ошибку с использованием таблицы синдромов. Убедиться  
в правильности полученного слова.  
**2.10. Сформировать кодовое слово длины n из слова длины k.** Внести  
двукратную ошибку в сформированное слово. Вычислить синдром,  
исправить ошибку с использованием таблицы синдромов. Убедиться  
в правильности полученного слова.  
**2.1. Сформировать кодовое слово длины n из слова длины k.** Внести  
трёхкратную ошибку в сформированное слово. Вычислить синдром,  
исправить ошибку с использованием таблицы синдромов.  
Убедиться, что полученное слово отличается от отправленного.  

# Лабораторная №3  

**3.1** Написать функцию формирования порождающей и проверочной  
матриц кода Хэмминга (𝟐𝒓 − 𝟏, 𝟐𝒓 − 𝒓 − 𝟏, 𝟑) на основе параметра 𝒓,  
а также таблицы синдромов для всех однократных ошибок.  
**3.2.** Провести исследование кода Хэмминга для одно-, двух- и  
трёхкратных ошибок для 𝒓 = 𝟐, 𝟑, 𝟒.  
**3.3**Написать функцию формирования порождающей и проверочной  
матриц расширенного кода Хэмминга (𝟐𝒓, 𝟐𝒓 − 𝒓 − 𝟏, 𝟑) на основе  
параметра 𝒓, а также таблицы синдромов для всех однократных  
ошибок. 
**3.4.** Провести исследование расширенного кода Хэмминга для одно-,  
двух-, трёх- и четырёхкратных ошибок для 𝒓 = 𝟐, 𝟑, 𝟒. 

# Лабораторная №4

**Часть 1**  
**4.1** Написать функцию формирования порождающей и проверочной  
матриц расширенного кода Голея (24,12,8).  
**4.2.** Провести исследование расширенного кода Голея для одно-, двух-,  
трёх- и четырёхкратных ошибок.  

**Часть 2**  
**4.3** Написать функцию формирования порождающей и проверочных  
матриц кода Рида-Маллера 𝑅𝑀(𝑟, 𝑚) на основе параметров 𝑟 и 𝑚.  
**4.4.** Провести исследование кода Рида-Маллера 𝑅𝑀(1,3) для одно- и  
двукратных ошибок.  
**4.5.** Провести исследование кода Рида-Маллера 𝑅𝑀(1,4) для одно-, двух-,  
трёх- и четырёхкратных ошибок.  

# Лабораторная №5
**Часть 1**  
**4.1** Написать функцию формирования порождающей матрицы кода РидаМаллера (r,m)  
в каноническом виде для произвольно заданных r и m.  
**4.2.** Реализовать алгоритм мажоритарного декодирования для кода РидаМаллера.  
**4.3.** Провести экспериментальную проверку алгоритма декодирования для  
кода Рида-Маллера (2,4).  


