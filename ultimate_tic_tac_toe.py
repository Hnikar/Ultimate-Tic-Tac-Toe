from __future__ import annotations
import sys
import enum
from typing import Union
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    BOARD_SIZE: int = 3
    WIN_REWARD: float = 1.0
    DRAW_REWARD: float = 0.0
    MAX_DEPTH: int = 2


class UltimateBoard:

    def __init__(self) -> None:
        self.board: list[list[GameBoard]] = [
            [GameBoard() for _ in range(Config.BOARD_SIZE)] for _ in range(Config.BOARD_SIZE)
        ]
        self.next_board: tuple[int, int] | None = None

    def possible_moves(self) -> list[tuple[int, int]]:
        if self.next_board is not None and not self.board[self.next_board[0]][self.next_board[1]].finished():
            row, col = self.next_board
            moves: list[tuple[int, int]] = self.board[row][col].possible_moves()
            return [
                (row * Config.BOARD_SIZE + r, col * Config.BOARD_SIZE + c) for r, c in moves
            ]

        all_moves: list[tuple[int, int]] = []
        for row in range(Config.BOARD_SIZE):
            for col in range(Config.BOARD_SIZE):
                moves: list[tuple[int, int]] = self.board[row][col].possible_moves()
                moves_converted: list[tuple[int, int]] = [
                    (row * Config.BOARD_SIZE + r, col * Config.BOARD_SIZE + c) for r, c in moves
                ]
                all_moves += moves_converted
        return all_moves

    def win(self, player: Player) -> bool:
        return any(
            all(self.board[i][j].win(player) for j in range(Config.BOARD_SIZE)) or
            all(self.board[j][i].win(player) for j in range(Config.BOARD_SIZE))
            for i in range(Config.BOARD_SIZE)
        ) or all(self.board[i][i].win(player) for i in range(Config.BOARD_SIZE)) or all(self.board[i][2 - i].win(player) for i in range(Config.BOARD_SIZE))

    def draw(self) -> bool:
        return not self.win(Player.PLAYER) and not self.win(Player.OPPONENT) and not self.possible_moves()

    def make_move(self, row: int, col: int, player: Player) -> None:
        board_row: int = int(row / 3)
        small_board_row: int = row % 3
        board_col: int = int(col / 3)
        small_board_col: int = col % 3
        self.board[board_row][board_col].make_move(small_board_row, small_board_col, player)
        self.next_board = (small_board_row, small_board_col)

    def set_empty(self, row: int, col: int) -> None:
        board_row: int = int(row / 3)
        small_board_row: int = row % 3
        board_col: int = int(col / 3)
        small_board_col: int = col % 3
        self.board[board_row][board_col].set_empty(small_board_row, small_board_col)


class Player(enum.Enum):
    OPPONENT = enum.auto()
    PLAYER = enum.auto()

    def opponent(self) -> Player:
        return Player.OPPONENT if self == Player.PLAYER else Player.PLAYER


class Cell(enum.Enum):
    EMPTY = enum.auto()


CellState = Union[Cell, Player]


class GameBoard:
    def __init__(self) -> None:
        self.board: list[list[CellState]] = [
            [Cell.EMPTY for _ in range(Config.BOARD_SIZE)] for _ in range(Config.BOARD_SIZE)
        ]

    def possible_moves(self) -> list[tuple[int, int]]:
        if self.win(Player.OPPONENT) or self.win(Player.PLAYER):
            return []
        return [
            (r, c)
            for r in range(Config.BOARD_SIZE)
            for c in range(Config.BOARD_SIZE)
            if self.board[r][c] == Cell.EMPTY
        ]

    def win(self, player: Player) -> bool:
        return any(
            all(self.board[i][j] == player for j in range(Config.BOARD_SIZE)) or
            all(self.board[j][i] == player for j in range(Config.BOARD_SIZE))
            for i in range(Config.BOARD_SIZE)
        ) or all(self.board[i][i] == player for i in range(Config.BOARD_SIZE)) or all(self.board[i][2 - i] == player for i in range(Config.BOARD_SIZE))

    def draw(self) -> bool:
        return not self.win(Player.PLAYER) and not self.win(Player.OPPONENT) and not self.possible_moves()
    
    def finished(self) -> bool:
        return self.win(Player.PLAYER) or self.win(Player.OPPONENT) or self.draw()

    def make_move(self, row: int, col: int, player: Player) -> None:
        if self.board[row][col] != Cell.EMPTY:
            raise ValueError("Cell is already taken")
        self.board[row][col] = player

    def set_empty(self, row: int, col: int) -> None:
        self.board[row][col] = Cell.EMPTY


def score(game_board: UltimateBoard) -> float:
    return 0.0


def minimax(game_board: UltimateBoard, player: Player, alpha: float, beta: float, depth: int) -> tuple[float, tuple[int, int] | None]:
    pass


def game_loop() -> None:
    game_board: UltimateBoard = UltimateBoard()
    while True:
        opponent_row, opponent_col = [int(i) for i in input().split()]
        valid_action_count = int(input())

        valid_moves: list[tuple[int, int]] = []
        for _ in range(valid_action_count):
            row, col = [int(j) for j in input().split()]
            valid_moves.append((row, col))

        if opponent_row != -1:
            game_board.make_move(opponent_row, opponent_col, Player.OPPONENT)

        reward, move = minimax(game_board, Player.PLAYER, -float("inf"), float("inf"), depth=Config.MAX_DEPTH)
        print(str(reward) + ' ' + str(move), game_board.possible_moves(), game_board.next_board, file=sys.stderr, flush=True)

        row, col = move
        game_board.make_move(row, col, Player.PLAYER)

        print(str(row) + ' ' + str(col), file=sys.stderr, flush=True)
        print(str(row) + ' ' + str(col))


game_loop()
