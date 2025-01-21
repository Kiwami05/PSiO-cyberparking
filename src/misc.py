"""
Moduł: misc.py
Opis: Funkcje bliżej nieokreślone.
Utworzono: 21-01-2025
"""


def intersection_over_union(rect1, rect2):
    """
    Oblicza stosunek przecięcia dwóch regionów/prostokątów.

    :param rect1: Krotka (x, y, w, h) reprezentująca pierwszy prostokąt o szerokości w i długości h, zaczynający się we współrzędnych x i y.
    :param rect2: Krotka (x, y, w, h) reprezentująca pierwszy prostokąt o szerokości w i długości h, zaczynający się we współrzędnych x i y.
    :return: Stosunek przecięcia pierwszego i drugiego prostokąta w skali [0, 1].
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height
    rect1_area = w1 * h1
    if rect1_area == 0:
        return 0
    return intersection_area / rect1_area
