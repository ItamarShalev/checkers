from __future__ import annotations

class Position:

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def right_direction(self, up: bool, count: int = 1):
        return self.right_up(count) if up else self.right_down(count)

    def left_direction(self, up: bool, count: int = 1):
        return self.left_up(count) if up else self.left_down(count)

    def right_down(self, count: int = 1):
        return Position(self.row + count, self.col + count)

    def right_up(self, count: int = 1):
        return Position(self.row - count, self.col + count)

    def left_down(self, count: int = 1):
        return Position(self.row + count, self.col - count)

    def left_up(self, count: int = 1):
        return Position(self.row - count, self.col - count)

    def count_to(self, other: Position):
        return abs(self.row - other.row)

    def is_valid(self):
        return 0 <= self.row < 8 and 0 <= self.col < 8

    def __eq__(self, other: object):
        if not isinstance(other, Position):
            return False
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return f"({self.row}, {self.col})"
