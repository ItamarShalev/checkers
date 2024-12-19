from abc import ABC, abstractmethod
from pathlib import Path

import pytest
import yaml

from src.data.board import Board
from src.data.enums import State
from src.data.piece import Piece
from src.data.position import Position
from src.data.step import Step, Steps
from src.data import exceptions
from src.game import Game, GameAI


class Action(ABC):

    @abstractmethod
    def __call__(self, game: Game):
        pass


class MoveAI(Action):

    def __init__(self, depth: int | None = None):
        self.depth = depth

    def __call__(self, game: Game):
        if not isinstance(game, GameAI):
            raise ValueError("The game is not an AI game.")
        game.move_ai(self.depth)


class Move(Action):

    def __init__(
            self,
            steps: dict | None = None,
            from_pos: list[int] | None = None,
            to_pos: list[int] | None = None,
            error: str | None = None):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.error = error
        if steps:
            self.steps = Steps([Step(Position(*step["from_pos"]), Position(*step["to_pos"])) for step in steps])
        elif from_pos and to_pos:
            self.steps = Steps(Position(*from_pos), Position(*to_pos))
        else:
            raise ValueError("Invalid arguments for Move action")

    def __call__(self, game: Game):
        if self.error:
            exception = getattr(exceptions, to_upper(self.error))
            with pytest.raises(exception):
                game.move(self.steps)
        else:
            game.move(self.steps)


class CheckBoard(Action):

    def __init__(self, excepted: list[list[str]]):
        self.excepted_board: Board = Board(excepted)

    def __call__(self, game: Game):
        assert game.board == self.excepted_board, "ERROR: The Board is not the same."


class CheckState(Action):

    def __init__(self, state: str):
        self.excepted_state: State = State.from_str(state)

    def __call__(self, game: Game):
        assert game.state() == self.excepted_state, "ERROR: The State is not the same."


class CheckPosition(Action):

    def __init__(self, positions: list[dict]):
        self.positions = positions

    def __call__(self, game: Game):
        for position_data in self.positions:
            pos = position_data["position"]
            piece = Piece.from_symbol(position_data["piece"])
            assert game.board[pos] == piece, f"ERROR: The piece at {pos} is not {piece}."


def all_yaml_files() -> list[Path]:
    yaml_dir = Path(__file__).parent / "yaml_test_case"
    return list(yaml_dir.rglob("*.yaml"))


def load_test_case(yaml_file: Path):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def to_upper(value: str) -> str:
    return value.replace("_", " ").title().replace(" ", "").replace("Ai", "AI")


@pytest.mark.parametrize("yaml_file", all_yaml_files(), ids=[file.name for file in all_yaml_files()])
def test_yaml_test_case(yaml_file: Path):
    data = load_test_case(yaml_file)
    game: Game = GameAI(**data["game_ai"]) if "game_ai" in data else Game(**data.get("game", {}))

    for index, action_data in enumerate(data.get("actions", [])):
        print(f"Start action: {index + 1}")
        print("Action data:", action_data)
        action_type = to_upper(action_data["type"])
        del action_data["type"]
        action = globals()[action_type](**action_data)
        action(game)
