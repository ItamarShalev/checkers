from __future__ import annotations

from enum import Enum


class Color(Enum):
    BLACK = "BLACK"
    WHITE = "WHITE"
    NO_COLOR = "NO_COLOR"

    @classmethod
    def from_str(cls, value: str | Color):
        if not value:
            return None
        if isinstance(value, Color):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid Color value: {value}") from None


class PieceSymbol(Enum):
    BLACK_KING = "BK"
    WHITE_KING = "WK"
    BLACK_PAWN = "B"
    WHITE_PAWN = "W"
    EMPTY = "_"

    def is_empty(self) -> bool:
        return self is PieceSymbol.EMPTY

    @property
    def color(self) -> Color:
        if self.is_empty():
            return Color.NO_COLOR
        return Color.BLACK if self in (PieceSymbol.BLACK_PAWN, PieceSymbol.BLACK_KING) else Color.WHITE

    @property
    def is_king(self) -> bool:
        return self in (PieceSymbol.BLACK_KING, PieceSymbol.WHITE_KING)

    @classmethod
    def from_str(cls, value: str | PieceSymbol):
        if isinstance(value, PieceSymbol):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            pass

        for member in cls:
            if member.value.upper() == value.upper():
                return member

        raise ValueError(f"Invalid piece value or key: {value}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class State(Enum):
    WHITE_TURN = "WHITE_TURN"
    BLACK_TURN = "BLACK_TURN"
    WHITE_WIN = "WHITE_WIN"
    BLACK_WIN = "BLACK_WIN"

    def is_finished(self) -> bool:
        return self in (State.WHITE_WIN, State.BLACK_WIN)

    @classmethod
    def from_str(cls, value: str | State):
        if not value:
            return None
        if isinstance(value, State):
            return value
        try:
            return cls[value.replace(" ", "_").upper()]
        except KeyError:
            raise ValueError(f"Invalid State value: {value}") from None
