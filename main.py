"""Венгерский алгоритм для полного паросочетания в двудольном графе."""


import argparse
from pathlib import Path


def read_matrix(path):
    """Первая строка: n или n m. Далее n строк по m весов; x — нет ребра."""
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("Файл пуст.")

    sizes = list(map(int, lines[0].split()))
    if len(sizes) == 1:
        n = m = sizes[0]
    elif len(sizes) == 2:
        n, m = sizes
    else:
        raise ValueError("В первой строке нужно указать n или n m.")
    if n < 1 or m < 1:
        raise ValueError("Размеры долей должны быть положительными.")
    if len(lines) != n + 1:
        raise ValueError(f"После размеров должно быть ровно {n} строк.")

    matrix = []
    for line in lines[1:]:
        row = [None if token.lower() == "x" else int(token)
               for token in line.split()]
        if len(row) != m:
            raise ValueError(f"В каждой строке должно быть ровно {m} элементов.")
        matrix.append(row)
    return matrix


def reduce_matrix(matrix):
    """Вычитает минимумы из строк, затем из столбцов."""
    n = len(matrix)
    a = [row[:] for row in matrix]

    # 1. Вычитаем минимум из каждой строки.
    for i in range(n):
        row_min = min(a[i])
        for j in range(n):
            a[i][j] -= row_min

    # 2. Вычитаем минимум из каждого столбца.
    for j in range(n):
        col_min = min(a[i][j] for i in range(n))
        for i in range(n):
            a[i][j] -= col_min

    return a


def maximum_zero_matching(matrix):
    """Находит максимальное паросочетание только по нулевым элементам матрицы."""
    n = len(matrix)
    match_col = [-1] * n

    def dfs(row, used_cols):
        for col in range(n):
            if matrix[row][col] != 0 or used_cols[col]:
                continue

            used_cols[col] = True

            if match_col[col] == -1 or dfs(match_col[col], used_cols):
                match_col[col] = row
                return True

        return False

    for row in range(n):
        dfs(row, [False] * n)

    match_row = [-1] * n
    for col, row in enumerate(match_col):
        if row != -1:
            match_row[row] = col

    return match_row, match_col


def minimum_zero_cover(matrix, match_row, match_col):
    """Строит минимальное покрытие всех нулей строками и столбцами.

    Используется теорема Кёнига:
    минимальное вершинное покрытие двудольного графа имеет тот же размер,
    что и максимальное паросочетание.
    """
    n = len(matrix)

    visited_rows = [False] * n
    visited_cols = [False] * n

    # Начинаем с непокрытых паросочетанием строк.
    stack = []

    for row in range(n):
        if match_row[row] == -1:
            visited_rows[row] = True
            stack.append(row)

    # Идём по чередующимся путям:
    # строка -> нулевое НЕ выбранное ребро -> столбец ->
    # выбранное ребро -> строка.
    while stack:
        row = stack.pop()

        for col in range(n):
            if matrix[row][col] != 0:
                continue

            if match_row[row] == col:
                continue

            if visited_cols[col]:
                continue

            visited_cols[col] = True

            matched_row = match_col[col]

            if matched_row != -1 and not visited_rows[matched_row]:
                visited_rows[matched_row] = True
                stack.append(matched_row)

    # Минимальное покрытие:
    # (непосещённые строки) U (посещённые столбцы)
    covered_rows = [not visited for visited in visited_rows]
    covered_cols = visited_cols[:]

    return covered_rows, covered_cols


def modify_matrix(matrix, covered_rows, covered_cols):
    """Выполняет модификацию матрицы по венгерскому алгоритму."""
    n = len(matrix)

    uncovered = [
        matrix[i][j]
        for i in range(n)
        for j in range(n)
        if not covered_rows[i] and not covered_cols[j]
    ]

    if not uncovered:
        raise RuntimeError("Нет незачёркнутых элементов для модификации.")

    minimum = min(uncovered)

    for i in range(n):
        for j in range(n):
            # Незачёркнутые элементы уменьшаем.
            if not covered_rows[i] and not covered_cols[j]:
                matrix[i][j] -= minimum

            # На пересечении двух линий прибавляем.
            elif covered_rows[i] and covered_cols[j]:
                matrix[i][j] += minimum

    return minimum


def hungarian_min(weights):
    """Минимальное по весу полное паросочетание."""
    n = len(weights)

    if any(len(row) != n for row in weights):
        raise ValueError("Матрица должна быть квадратной.")

    matrix = reduce_matrix(weights)

    while True:
        match_row, match_col = maximum_zero_matching(matrix)

        # Если есть n независимых нулей, полное паросочетание найдено.
        if all(col != -1 for col in match_row):
            total = sum(weights[row][match_row[row]] for row in range(n))
            return match_row, total

        covered_rows, covered_cols = minimum_zero_cover(
            matrix,
            match_row,
            match_col,
        )

        modify_matrix(matrix, covered_rows, covered_cols)


def hungarian(weights, mode="min"):
    """Возвращает (пары, вес) либо None, если полного паросочетания нет.

    Полное паросочетание покрывает ВСЕ вершины обеих долей.
    None в матрице означает отсутствующее ребро.
    """
    if mode not in ("min", "max"):
        raise ValueError("Режим должен быть min или max.")
    if not weights or not weights[0]:
        raise ValueError("Матрица не должна быть пустой.")
    n, m = len(weights), len(weights[0])
    if any(len(row) != m for row in weights):
        raise ValueError("Строки матрицы должны иметь одинаковую длину.")
    if any(value is not None and not isinstance(value, int)
           for row in weights for value in row):
        raise ValueError("Вес должен быть целым числом или None.")
    if n != m:
        return None

    values = [value for row in weights for value in row if value is not None]
    if not values:
        return None
    low, high = min(values), max(values)

    # Стоимости существующих рёбер лежат в [0, high - low].
    # Поэтому штраф больше стоимости любого допустимого полного паросочетания.
    penalty = n * (high - low) + 1
    costs = []
    for row in weights:
        costs.append([
            penalty if value is None else
            (value - low if mode == "min" else high - value)
            for value in row
        ])

    matching, _ = hungarian_min(costs)
    if any(weights[row][col] is None for row, col in enumerate(matching)):
        return None
    total = sum(weights[row][col] for row, col in enumerate(matching))
    return matching, total


def print_result(weights, matching, total, mode):
    """Печатает найденное паросочетание."""
    title = "минимальное" if mode == "min" else "максимальное"

    print(f"Найдено {title} полное паросочетание:")
    print()

    for row, col in enumerate(matching):
        print(
            f"L{row + 1} -- R{col + 1}, "
            f"вес = {weights[row][col]}"
        )

    print()
    print(f"Суммарный вес: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Венгерский алгоритм для двудольного графа."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="graph.txt",
        help="Файл с матрицей весов.",
    )

    parser.add_argument(
        "--mode",
        choices=["min", "max"],
        default="min",
        help="Искать минимум или максимум.",
    )

    args = parser.parse_args()

    try:
        weights = read_matrix(args.input)
        result = hungarian(weights, args.mode)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")

    if result is None:
        print("Полного паросочетания в графе не существует.")
        return

    matching, total = result
    print_result(weights, matching, total, args.mode)


if __name__ == "__main__":
    main()
