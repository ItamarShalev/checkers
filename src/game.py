import sys

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
        self._board.move_piece(steps)
        self._current_turn = Color.BLACK if self._current_turn == Color.WHITE else Color.WHITE
        return True

    def state(self) -> State:
        if not self._board.has_piece(Color.WHITE):
            return State.BLACK_WIN
        if not self._board.has_piece(Color.BLACK):
            return State.WHITE_WIN
        return State.WHITE_TURN if self._current_turn == Color.WHITE else State.BLACK_TURN


class GameAI(Game):

    KING_WEIGHT = 4
    MAX_INT = sys.maxsize
    MIN_INT = -sys.maxsize - 1
    DEPTH = 3

    def __init__(
            self,
            board: list[list[str]] | None = None,
            current_turn: str | Color = Color.WHITE,
            ai_color: str | Color = Color.BLACK,
            depth: int = DEPTH):
        super().__init__(board, current_turn)
        self._ai_color: Color = Color.from_str(ai_color)
        self._depth: int = depth


    def move_ai(self, depth: int | None = None) -> bool:
        # The evaluation function treat white as maximizing player
        is_maximizing = self._ai_color is Color.WHITE
        _, best_move = self._minimax(self._board, depth or self._depth, GameAI.MIN_INT, GameAI.MAX_INT, is_maximizing)
        if not best_move:
            raise InvalidMove("AI could not find a valid move")
        self.move(best_move)
        return True

    def _minimax(self,
                 board: Board,
                 depth: int,
                 alpha: int,
                 beta: int,
                 is_maximizing: bool) -> tuple[int, Steps | None]:
        if depth == 0 or not board.has_piece(Color.WHITE) or not board.has_piece(Color.BLACK):
            return self._evaluate_board(board), None

        if is_maximizing:
            max_eval = GameAI.MIN_INT
            best_move = None
            for move in self._get_all_possible_moves(Color.WHITE):
                new_board = board.copy()
                new_board.move_piece(move)
                evaluation, _ = self._minimax(new_board, depth - 1, alpha, beta, False)
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval, best_move

        min_eval = GameAI.MAX_INT
        best_move = None
        for move in self._get_all_possible_moves(Color.BLACK):
            new_board = board.copy()
            new_board.move_piece(move)
            evaluation, _ = self._minimax(new_board, depth - 1, alpha, beta, True)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval, best_move

    def _get_all_possible_moves(self, color: Color) -> list[Steps]:
        all_moves = []
        pieces = self._board.pieces(color)
        for piece_pos in pieces:
            piece = self._board[piece_pos]
            all_moves.extend(piece.all_possible_moves(piece_pos, self._board))
        return all_moves

    def _evaluate_board(self, board: Board) -> int:
        black_win = not board.has_piece(Color.WHITE)
        white_win = not board.has_piece(Color.BLACK)

        if black_win:
            return GameAI.MIN_INT
        if white_win:
            return GameAI.MAX_INT

        black_pieces = len(board.pieces(Color.BLACK))
        black_king_pieces = len(board.king_pieces(Color.BLACK))
        white_pieces = len(board.pieces(Color.WHITE))
        white_king_pieces = len(board.king_pieces(Color.WHITE))
        total_black_score = black_pieces + GameAI.KING_WEIGHT * black_king_pieces
        total_white_score = white_pieces + GameAI.KING_WEIGHT * white_king_pieces
        return total_white_score - total_black_score
