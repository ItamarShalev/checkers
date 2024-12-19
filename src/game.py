from src.data.board import Board
from src.data.enums import Color, PieceSymbol, State
from src.data.exceptions import InvalidMove, InvalidTurn
from src.data.step import Steps


class Game:

    def __init__(self, board: list[list[str]] | None = None, current_turn: str | Color = Color.WHITE):
        self._board: Board = Board(board)
        self._current_turn: Color = Color.from_str(current_turn)


    @property
    def board(self) -> Board:
        return self._board


    def _validate_move(self, steps: Steps) -> list[Steps]:
        from_pos = steps[0].from_pos
        from_piece = self._board[from_pos]
        if from_piece.symbol is PieceSymbol.EMPTY:
            raise InvalidMove(f"Invalid move empty spot from {from_pos} to {steps[-1].to_pos}")
        if from_piece.color != self._current_turn:
            raise InvalidTurn(f"Invalid turn for color: {from_piece.color}, expected: {self._current_turn}")
        all_possible_moves = from_piece.all_possible_moves(from_pos, self._board)
        if steps not in all_possible_moves:
            print(self.board)
            raise InvalidMove(f"Invalid move from {from_pos} to {steps[-1].to_pos}")
        return all_possible_moves

    def move(self, steps: Steps) -> bool:
        all_possible_moves = self._validate_move(steps)
        steps = all_possible_moves[all_possible_moves.index(steps)]
        for step in steps:
            self._board.move_piece(step)
        self._current_turn = Color.BLACK if self._current_turn == Color.WHITE else Color.WHITE
        return True

    def state(self) -> State:
        if not self._board.has_piece(Color.WHITE):
            return State.BLACK_WIN
        if not self._board.has_piece(Color.BLACK):
            return State.WHITE_WIN
        return State.WHITE_TURN if self._current_turn == Color.WHITE else State.BLACK_TURN
