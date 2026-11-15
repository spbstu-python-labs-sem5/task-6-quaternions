import pytest

from quaternion import Quaternion


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ((1, 2, 3, 4), (5, 6, 7, 8), (6, 8, 10, 12)),
        ((1, -2, 3.5, 4), (0, 0, 0, 0), (1, -2, 3.5, 4)),
        ((1, -2, 3, -4), (-1, 2, -3, 4), (0, 0, 0, 0)),
    ],
)
def test_addition_is_componentwise_and_preserves_operands(left, right, expected):
    first = Quaternion(*left)
    second = Quaternion(*right)

    result = first + second

    assert isinstance(result, Quaternion)
    assert (result.w, result.x, result.y, result.z) == expected

    # проверка на то что в результате сложения создается новый объект
    assert result is not first
    assert result is not second
    assert (first.w, first.x, first.y, first.z) == left
    assert (second.w, second.x, second.y, second.z) == right


@pytest.mark.parametrize(
    "components, expected",
    [
        ((0, 0, 0, 0), 0),
        ((1, 0, 0, 0), 1),
        ((0, 0, -1, 0), 1),
        ((1, 2, 2, 4), 5),
        ((-1, -2, -2, -4), 5),
        ((0.5, 0.5, 0.5, 0.5), 1),
    ],
)
def test_norm(components, expected):
    assert Quaternion(*components).norm() == pytest.approx(expected)
