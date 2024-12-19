from __future__ import annotations

import copy

from src.data.enums import PieceSymbol, Color
from src.data.exceptions import InvalidBoard
from src.data.piece import Piece, EmptySpot, King
from src.data.position import Position
from src.data.step import Step


class Board:
    """Represents the checkers board."""

    def __init__(self, board: list[list[str]] | None = None):
        self._board: list[list[Piece]] = Board.create_board(board)

    @staticmethod
    def default_board() -> list[list[str]]:
        board = [
            ["B", "_", "B", "_", "B", "_", "B", "_"],
            ["_", "B", "_", "B", "_", "B", "_", "B"],
            ["B", "_", "B", "_", "B", "_", "B", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "W", "_", "W", "_", "W", "_", "W"],
            ["W", "_", "W", "_", "W", "_", "W", "_"],
            ["_", "W", "_", "W", "_", "W", "_", "W"]
        ]
        return board

    @staticmethod
    def validate_board(board: list[list[Piece]]):
        size_correct = all(len(row) == 8 for row in board) and len(board) == 8
        valid_number_color_pieces = sum(piece.color == Color.BLACK for row in board for piece in row) <= 12
        valid_number_white_pieces = sum(piece.color == Color.BLACK for row in board for piece in row) <= 12
        there_is_piece = any(piece.symbol != PieceSymbol.EMPTY for row in board for piece in row)

        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1 and board[row][col].symbol != PieceSymbol.EMPTY:
                    raise InvalidBoard(f"Invalid piece at position ({row}, {col})")
        if not there_is_piece:
            raise InvalidBoard("There must be at least one piece on the board.")
        if not size_correct:
            raise InvalidBoard("Invalid board size")
        if not valid_number_white_pieces:
            raise InvalidBoard("Invalid number of white pieces, must be less than 12.")
        if not valid_number_color_pieces:
            raise InvalidBoard("Invalid number of black pieces, must be less than 12.")

    @staticmethod
    def create_board(board_data: list[list[str]] | None = None) -> list[list[Piece]]:
        board_data = board_data or Board.default_board()
        board = [[Piece.from_symbol(symbol) for symbol in row] for row in board_data]
        Board.validate_board(board)
        return board

    def pieces(self, color: Color) -> list[Position]:
        return [Position(row, col) for row in range(8) for col in range(8) if self._board[row][col].color == color]

    def has_piece(self, color: Color) -> bool:
        return any(self._board[row][col].color == color for row in range(8) for col in range(8))

    def __getitem__(self, position: list[int] | Position) -> Piece:
        if isinstance(position, list):
            position = Position(*position)
        if isinstance(position, Position):
            return self._board[position.row][position.col]
        raise ValueError("Invalid key type")

    def __setitem__(self, position: list[int] | Position, piece: Piece):
        if isinstance(piece, PieceSymbol):
            piece = Piece.from_symbol(piece)
        if isinstance(position, list):
            position = Position(*position)
        self._board[position.row][position.col] = piece

    def __copy__(self) -> Board:
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> Board:
        new_board = Board()
        new_board._board = copy.deepcopy(self._board, memo)
        return new_board

    def __eq__(self, other: object):
        if not isinstance(other, Board):
            return False
        return self._board == other._board

    def __hash__(self):
        return hash(tuple(tuple(hash(piece) for piece in row) for row in self._board if row))

    def __repr__(self):
        return "\n".join([" ".join([str(piece.symbol.value) for piece in row]) for row in self._board])

    def copy(self):
        return copy.deepcopy(self)

    def move_piece(self, step: Step):
        self[step.to_pos] = self[step.from_pos]
        self[step.from_pos] = EmptySpot()
        if step.eaten_piece:
            self[step.eaten_piece] = EmptySpot()

        is_pawn = self[step.to_pos].symbol in [PieceSymbol.WHITE_PAWN, PieceSymbol.BLACK_PAWN]
        need_promotion = step.to_pos.row == 0 if self[step.to_pos].color is Color.WHITE else step.to_pos.row == 7
        if is_pawn and need_promotion:
            self[step.to_pos] = King(self[step.to_pos].color)
