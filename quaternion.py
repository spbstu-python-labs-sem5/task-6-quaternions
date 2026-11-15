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

    def norm(self) -> float:
        """Евклидова норма кватерниона"""
        # hypot - евклидова норма с любым числом аргументов
        return math.hypot(self.w, self.x, self.y, self.z)
