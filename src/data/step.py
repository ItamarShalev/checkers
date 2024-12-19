from __future__ import annotations

from typing import Optional

from src.data.position import Position


class Step:
    def __init__(self, from_pos: Position, to_pos: Position, eaten_piece: Optional[Position] = None):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.eaten_piece = eaten_piece

    def __eq__(self, other: object):
        if not isinstance(other, Step):
            return False
        return self.from_pos == other.from_pos and self.to_pos == other.to_pos

    def __repr__(self):
        return f"Step({self.from_pos} -> {self.to_pos})"


class Steps:
    def __init__(self,
            steps: list[Step] | Position,
            to_pos: Position | None = None,
            eaten_piece: Position | None = None):
        if isinstance(steps, list):
            self.steps = steps
        elif isinstance(steps, Position) and to_pos is not None:
            self.steps = [Step(steps, to_pos, eaten_piece)]
        else:
            raise ValueError("Invalid arguments for append method")

    def append(self, step: Step):
        self.steps.append(step)

    def __eq__(self, other: object):
        if not isinstance(other, Steps):
            return False
        if len(self.steps) != len(other):
            return False
        return all(step == other_step for step, other_step in zip(self.steps, other))

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, key: int):
        return self.steps[key]

    def __len__(self):
        return len(self.steps)

    def __str__(self):
        return f"Steps({" -> ".join([f"{step.from_pos} -> {step.to_pos}" for step in self.steps])})"

    def __add__(self, other):
        return Steps(self.steps + other.steps)

    def __iadd__(self, other):
        self.steps += other.steps
        return self

    def __radd__(self, other):
        if not other:
            return self
        return Steps(other + self.steps)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(tuple(self.steps))
