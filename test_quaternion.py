import math

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


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),  # i*j = k
        ((0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, -1)),  # j*i = -k
        ((0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0)),  # j*k = i
        ((0, 0, 0, 1), (0, 0, 1, 0), (0, -1, 0, 0)),  # k*j = -i
        ((0, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0)),  # k*i = j
        ((0, 1, 0, 0), (0, 0, 0, 1), (0, 0, -1, 0)),  # i*k = -j
        ((0, 1, 0, 0), (0, 1, 0, 0), (-1, 0, 0, 0)),  # i*i = -1
        ((0, 0, 1, 0), (0, 0, 1, 0), (-1, 0, 0, 0)),  # j*j = -1
        ((0, 0, 0, 1), (0, 0, 0, 1), (-1, 0, 0, 0)),  # k*k = -1
        ((1, 2, 3, 4), (5, 6, 7, 8), (-60, 12, 30, 24)),
        ((1, -2, 3, -4), (1, 0, 0, 0), (1, -2, 3, -4)),
        ((1, 0, 0, 0), (1, -2, 3, -4), (1, -2, 3, -4)),
        ((1, 2, 3, 4), (0, 0, 0, 0), (0, 0, 0, 0)),
    ],
)
def test_multiplication(left, right, expected):
    first = Quaternion(*left)
    second = Quaternion(*right)

    result = first * second

    assert isinstance(result, Quaternion)
    assert (result.w, result.x, result.y, result.z) == pytest.approx(expected)


@pytest.mark.parametrize(
    "components, expected",
    [
        ((1, 2, -3, 4), (1, -2, 3, -4)),
        ((-2, 0.5, -1.5, 0), (-2, -0.5, 1.5, 0)),
        ((0, 0, 0, 0), (0, 0, 0, 0)),
    ],
)
def test_conjugate(components, expected):
    q = Quaternion(*components)

    result = q.conjugate()

    assert (result.w, result.x, result.y, result.z) == expected

    restored = result.conjugate()
    assert (restored.w, restored.x, restored.y, restored.z) == components


@pytest.mark.parametrize(
    "components, expected",
    [
        ((1, 0, 0, 0), (1, 0, 0, 0)),
        ((0, 1, 0, 0), (0, -1, 0, 0)),
        ((2, 0, 0, 0), (0.5, 0, 0, 0)),
        ((1, 2, 3, 4), (1 / 30, -2 / 30, -3 / 30, -4 / 30)),
        ((-0.5, 0.5, -0.5, 0.5), (-0.5, -0.5, 0.5, -0.5)),
    ],
)
def test_inverse(components, expected):
    q = Quaternion(*components)

    inverse = q.inverse()

    assert (inverse.w, inverse.x, inverse.y, inverse.z) == pytest.approx(expected)

    # умножение на обратный квантернион дает единицу
    for product in (q * inverse, inverse * q):
        assert (product.w, product.x, product.y, product.z) == pytest.approx((1, 0, 0, 0))


def test_zero_quaternion_has_no_inverse():
    with pytest.raises(ZeroDivisionError, match="Нулевой кватернион"):
        Quaternion(0, 0, 0, 0).inverse()


# для поворота на 90 используются cos(45) = sin(45) = sqrt(0.5)
@pytest.mark.parametrize(
    "components, vector, expected",
    [
        # Вокруг X на +90: вектор вдоль положительной оси Y переходит в +Z
        ((math.sqrt(0.5), math.sqrt(0.5), 0, 0), (0, 1, 0), (0, 0, 1)),
        # Вокруг Y на +90: вектор вдоль положительной оси Z переходит в +X
        ((math.sqrt(0.5), 0, math.sqrt(0.5), 0), (0, 0, 1), (1, 0, 0)),
        # Тот же поворот вокруг Y, но исходный вектор направлен вдоль +X
        ((math.sqrt(0.5), 0, math.sqrt(0.5), 0), (1, 0, 0), (0, 0, -1)),
        # Вокруг Z на +90: вектор вдоль +X переходит в +Y
        ((math.sqrt(0.5), 0, 0, math.sqrt(0.5)), (1, 0, 0), (0, 1, 0)),
        # Вокруг Z на -90: направление поворота меняется, +X переходит в -Y
        ((math.sqrt(0.5), 0, 0, -math.sqrt(0.5)), (1, 0, 0), (0, -1, 0)),
        # Вокруг Z на 180: +X переходит в -X
        ((0, 0, 0, 1), (1, 0, 0), (-1, 0, 0)),
        # Вокруг Z на +90 для вектора с тремя ненулевыми компонентами
        ((math.sqrt(0.5), 0, 0, math.sqrt(0.5)), (2, -3, 4), (3, 2, 4)),
        # Вокруг оси (1, 1, 1) на +120: направления X -> Y -> Z -> X
        # После нормализации оси ее компоненты равны 1/sqrt(3),
        # а cos(60) = 0.5 и sin(60)/sqrt(3) = 0.5
        # Этот случай проверяет поворот вокруг оси, не совпадающей с X, Y или Z
        ((0.5, 0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0)),
    ],
)
def test_rotate_vector(components, vector, expected):
    q = Quaternion(*components)

    assert q.rotate_vector(vector) == pytest.approx(expected)


def test_rotation_composition():
    rotation_x = Quaternion(math.sqrt(0.5), math.sqrt(0.5), 0, 0)
    rotation_z = Quaternion(math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    vector = (1, 2, 3)

    # Сначала поворот вокруг X на +90: (1, 2, 3) -> (1, -3, 2)
    # Затем вокруг Z на +90: (1, -3, 2) -> (3, 1, 2)
    sequential = rotation_z.rotate_vector(rotation_x.rotate_vector(vector))

    # q = q_z * q_x.
    combined = rotation_z * rotation_x

    assert sequential == pytest.approx((3, 1, 2))
    assert combined.rotate_vector(vector) == pytest.approx(sequential)
