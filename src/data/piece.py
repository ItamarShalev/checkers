from abc import ABC, abstractmethod
from functools import partial

from src.data.enums import Color, PieceSymbol
from src.data.position import Position
from src.data.step import Steps

class Piece(ABC):

    def __init__(self, symbol: PieceSymbol):
        self.symbol: PieceSymbol = symbol
        self.color: Color = symbol.color
        self.is_king: bool = symbol.is_king

    @staticmethod
    def from_symbol(symbol: str | PieceSymbol):
        piece_symbol = PieceSymbol.from_str(symbol)
        piece_classes = {
            PieceSymbol.BLACK_PAWN: lambda: Pawn(Color.BLACK),
            PieceSymbol.WHITE_PAWN: lambda: Pawn(Color.WHITE),
            PieceSymbol.BLACK_KING: lambda: King(Color.BLACK),
            PieceSymbol.WHITE_KING: lambda: King(Color.WHITE),
            PieceSymbol.EMPTY: EmptySpot
        }
        return piece_classes.get(piece_symbol, EmptySpot)()

    @abstractmethod
    def all_possible_moves(self, current_position: Position, board, pre_steps: Steps | None = None) -> list[Steps]:
        pass

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.symbol == other.symbol

class EmptySpot(Piece):

    def __init__(self):
        super().__init__(PieceSymbol.EMPTY)

    def all_possible_moves(self, current_position: Position, board, pre_steps: Steps | None = None) -> list[Steps]:
        return []

class MoveCalculator:

    MAX_STEPS = 8

    @staticmethod
    def calculate_moves(
            current_position: Position,
            board,
            pre_steps: Steps | None,
            directions: list,
            color: Color,
            max_steps: int = MAX_STEPS) -> list[Steps]:
        steps = []
        for direction in directions:
            for i in range(1, max_steps + 1):
                new_position = direction(i)

                if not new_position.is_valid():
                    break

                if board[new_position].color is color:
                    break

                if board[new_position].symbol is PieceSymbol.EMPTY:
                    next_steps = pre_steps + Steps(current_position, new_position)
                    steps.append(next_steps)

                if board[new_position].symbol is not PieceSymbol.EMPTY and board[new_position].color != color:
                    jump_position = direction(i + 1)
                    if jump_position.is_valid() and board[jump_position].symbol == PieceSymbol.EMPTY:
                        next_steps = pre_steps + Steps(current_position, jump_position, eaten_piece=new_position)
                        steps.append(next_steps)
                        new_board = board.copy()
                        new_board.move_piece(next_steps[-1])
                        piece = new_board[jump_position]
                        steps.extend(piece.all_possible_moves(jump_position, new_board, next_steps))
                    break
        return steps

class Pawn(Piece):

    def __init__(self, color: Color):
        symbol = PieceSymbol.WHITE_PAWN if color is Color.WHITE else PieceSymbol.BLACK_PAWN
        super().__init__(symbol)

    def all_possible_moves(self, current_position: Position, board, pre_steps: Steps | None = None) -> list[Steps]:
        up = self.color is Color.WHITE
        directions = [
            partial(current_position.right_direction, up),
            partial(current_position.left_direction, up),
        ]
        return MoveCalculator.calculate_moves(current_position, board, pre_steps, directions, self.color, max_steps=1)

    def needs_promotion(self, position: Position):
        return position.row == 0 if self.color is Color.WHITE else position.row == 7

    def promote(self):
        return King(self.color)

class King(Piece):

    def __init__(self, color: Color):
        symbol = PieceSymbol.WHITE_KING if color is Color.WHITE else PieceSymbol.BLACK_KING
        super().__init__(symbol)

    def all_possible_moves(self, current_position: Position, board, pre_steps: Steps | None = None) -> list[Steps]:
        directions = [
            partial(current_position.right_up),
            partial(current_position.left_up),
            partial(current_position.right_down),
            partial(current_position.left_down),
        ]
        return MoveCalculator.calculate_moves(current_position, board, pre_steps, directions, self.color)
