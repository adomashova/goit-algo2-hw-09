"""
Домашнє завдання до теми "Локальний пошук, евристики та імітація відпалу".

Мінімізація функції Сфери f(x) = sum(x_i^2), x_i in [-5, 5], трьома
методами локальної оптимізації:
  - hill_climbing        - алгоритм "Підйом на гору";
  - random_local_search  - випадковий локальний пошук;
  - simulated_annealing  - імітація відпалу.

Глобальний мінімум функції: f(0, ..., 0) = 0.

Запуск:
    python local_search.py
"""

import math
import random

STEP_SIZE = 0.1  # радіус пошуку сусідніх точок


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi ** 2 for xi in x)


def _clip(value, low, high):
    """Обмежує значення координати межами [low, high]."""
    return max(low, min(high, value))


def _random_point(bounds):
    """Випадкова точка в межах bounds."""
    return [random.uniform(low, high) for low, high in bounds]


def _random_neighbor(point, bounds, step=STEP_SIZE):
    """Випадковий сусід точки: кожна координата зсувається на величину
    з [-step, step] і обмежується межами."""
    return [
        _clip(xi + random.uniform(-step, step), low, high)
        for xi, (low, high) in zip(point, bounds)
    ]


def _distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    """Підйом на гору (для мінімізації - спуск): на кожній ітерації
    перебираємо сусідів (зсув кожної координати на +-step) і переходимо
    до найкращого з них, якщо він кращий за поточну точку. Якщо жоден
    сусід не кращий - зменшуємо крок удвічі (уточнюємо пошук).

    Завершується, коли: вичерпано iterations; крок став меншим за epsilon;
    або зміна значення функції / положення точки менша за epsilon.
    """
    current = _random_point(bounds)
    current_value = func(current)
    step = STEP_SIZE

    for _ in range(iterations):
        best_neighbor = None
        best_value = current_value

        for i, (low, high) in enumerate(bounds):
            for delta in (-step, step):
                neighbor = current.copy()
                neighbor[i] = _clip(neighbor[i] + delta, low, high)
                value = func(neighbor)
                if value < best_value:
                    best_neighbor, best_value = neighbor, value

        if best_neighbor is None:  # покращення немає - зменшуємо крок
            step /= 2
            if step < epsilon:
                break
            continue

        value_change = current_value - best_value
        position_change = _distance(current, best_neighbor)
        current, current_value = best_neighbor, best_value

        if value_change < epsilon or position_change < epsilon:
            break

    return current, current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    """Випадковий локальний пошук: на кожній ітерації генеруємо
    випадкового сусіда поточної точки й переходимо до нього, лише якщо
    він кращий.

    Завершується, коли вичерпано iterations або коли прийняте покращення
    (зміна значення функції) стає меншим за epsilon.
    """
    current = _random_point(bounds)
    current_value = func(current)

    for _ in range(iterations):
        candidate = _random_neighbor(current, bounds)
        candidate_value = func(candidate)

        if candidate_value < current_value:
            value_change = current_value - candidate_value
            current, current_value = candidate, candidate_value
            if value_change < epsilon:
                break

    return current, current_value


# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000,
                        cooling_rate=0.95, epsilon=1e-6):
    """Імітація відпалу: гіршого сусіда приймаємо з імовірністю
    exp(-delta / temp), де delta - погіршення значення функції. Із
    падінням температури (temp *= cooling_rate) ймовірність прийняття
    гірших розв'язків зменшується.

    Завершується, коли: вичерпано iterations або температура впала
    нижче epsilon (пошукова здатність вичерпана).
    """
    current = _random_point(bounds)
    current_value = func(current)
    best, best_value = current.copy(), current_value
    initial_temp = temp

    for _ in range(iterations):
        if temp < epsilon:
            break

        # крок пошуку зменшується разом із температурою: спочатку
        # "стрибаємо" далеко (дослідження), наприкінці - дрібно (уточнення)
        step = STEP_SIZE + math.sqrt(temp / initial_temp)
        candidate = _random_neighbor(current, bounds, step)
        candidate_value = func(candidate)
        delta = candidate_value - current_value

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current, current_value = candidate, candidate_value
            if current_value < best_value:
                best, best_value = current.copy(), current_value

        temp *= cooling_rate

    return best, best_value


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
