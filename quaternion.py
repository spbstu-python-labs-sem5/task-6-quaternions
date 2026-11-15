import math


class Quaternion:
    """Кватернион w + xi + yj + zk"""

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Quaternion: (w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def __add__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            return NotImplemented
        
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            return NotImplemented

        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def conjugate(self) -> "Quaternion":
        """Сопряженный кватернион: w - xi - yj - zk"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion":
        """Обратный кватернион: сопряжение, деленное на квадрат нормы"""
        norm = self.norm()
        if norm == 0:
            raise ZeroDivisionError("Нулевой кватернион не имеет обратного")

        norm_squared = norm * norm
        conjugate = self.conjugate()
        return Quaternion(
            conjugate.w / norm_squared,
            conjugate.x / norm_squared,
            conjugate.y / norm_squared,
            conjugate.z / norm_squared,
        )

    def norm(self) -> float:
        """Евклидова норма кватерниона"""
        # hypot - евклидова норма с любым числом аргументов
        return math.hypot(self.w, self.x, self.y, self.z)

    def rotate_vector(self, vector: tuple[float, float, float] ) -> tuple[float, float, float]:
        """Повернуть 3D-вектор по формуле q * (0, vx, vy, vz) * q^-1"""
        x, y, z = vector
        rotated = self * Quaternion(0, x, y, z) * self.inverse()
        return rotated.x, rotated.y, rotated.z
