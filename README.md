# Высокопроизводительные вычисления
## Бехтерев Алексей, группа 6133
____
# Лабораторная работа 0 (MatMul)
Лабораторная работа выполнена на языке Python. Использование CUDA достигается библиотекой Numba.
Выполнение операции (метода) по произведению матриц в параллельном режиме обеспечивает декоратор **@cuda.jit**, который производит вычисления на CUDA

В результате получается достичь значительного ускорения вычислений при использовании GPU в сравнении с использованием CPU.

Эксперименты с варьированием размерности матриц представлены в файле result.csv
