from __future__ import annotations
import sys
import enum
import time
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    BOARD_SIZE: int = 3
    TIME_LIMIT: float = 0.095
    WIN_REWARD: float = 10000.0
    SMALL_WIN_REWARD: float = 100.0

class Player(enum.Enum):
    OPPONENT = -1
    NONE = 0
    PLAYER = 1

    @property
    def opponent(self) -> Player:
        if self == Player.PLAYER: return Player.OPPONENT
        if self == Player.OPPONENT: return Player.PLAYER
        return Player.NONE

class Cell(enum.Enum):
    EMPTY = 0

CellState = Union[int, Player]

class GameBoard:
    def __init__(self) -> None:
        self.board: List[List[int]] = [[0] * 3 for _ in range(3)]
        self.winner: Player = Player.NONE
        self.filled_count: int = 0

    def make_move(self, row: int, col: int, player: Player) -> None:
        if self.board[row][col] == 0:
            self.board[row][col] = player.value
            self.filled_count += 1
            self.update_winner(row, col, player)

    def undo_move(self, row: int, col: int) -> None:
        self.board[row][col] = 0
        self.filled_count -= 1
        self.winner = Player.NONE

    def update_winner(self, last_r: int, last_c: int, player: Player) -> None:
        val = player.value
        b = self.board
        if (b[last_r][0] == val and b[last_r][1] == val and b[last_r][2] == val) or \
           (b[0][last_c] == val and b[1][last_c] == val and b[2][last_c] == val):
            self.winner = player
            return

        if last_r == last_c:
             if b[0][0] == val and b[1][1] == val and b[2][2] == val:
                self.winner = player
                return
        
        if last_r + last_c == 2:
            if b[0][2] == val and b[1][1] == val and b[2][0] == val:
                self.winner = player
                return

    def is_full(self) -> bool:
        return self.filled_count == 9

class UltimateBoard:
    def __init__(self) -> None:
        self.board: List[List[GameBoard]] = [
            [GameBoard() for _ in range(3)] for _ in range(3)
        ]
        self.next_board_coords: Tuple[int, int] = (-1, -1) 

    def make_move(self, row: int, col: int, player: Player) -> None:
        b_row, b_col = row // 3, col // 3
        s_row, s_col = row % 3, col % 3
        
        self.board[b_row][b_col].make_move(s_row, s_col, player)
        
        target_board = self.board[s_row][s_col]
        if target_board.winner != Player.NONE or target_board.is_full():
            self.next_board_coords = (-1, -1)
        else:
            self.next_board_coords = (s_row, s_col)

    def undo_move(self, row: int, col: int, prev_next_coords: Tuple[int, int]) -> None:
        b_row, b_col = row // 3, col // 3
        s_row, s_col = row % 3, col % 3
        self.board[b_row][b_col].undo_move(s_row, s_col)
        self.next_board_coords = prev_next_coords

class AI:
    def __init__(self):
        self.start_time = 0.0

    def evaluate(self, game: UltimateBoard) -> float:
        score = 0.0
        weights = [[1.5, 1.0, 1.5], [1.0, 2.0, 1.0], [1.5, 1.0, 1.5]] 
        
        for r in range(3):
            for c in range(3):
                sub = game.board[r][c]
                if sub.winner == Player.PLAYER:
                    score += Config.SMALL_WIN_REWARD * weights[r][c]
                elif sub.winner == Player.OPPONENT:
                    score -= Config.SMALL_WIN_REWARD * weights[r][c]
                else:
                    if sub.board[1][1] == Player.PLAYER.value:
                        score += 2 * weights[r][c]
                    elif sub.board[1][1] == Player.OPPONENT.value:
                        score -= 2 * weights[r][c]
        return score

    def minimax(self, game: UltimateBoard, depth: int, alpha: float, beta: float, 
                maximizing: bool, valid_moves_cache: Optional[List[Tuple[int, int]]] = None) -> float:
        
        if (depth % 2 == 0) and (time.time() - self.start_time > Config.TIME_LIMIT):
            raise TimeoutError()

        if depth == 0:
            return self.evaluate(game)

        moves = []
        if valid_moves_cache:
            moves = valid_moves_cache
        else:
            nb = game.next_board_coords
            if nb == (-1, -1):
                for br in range(3):
                    for bc in range(3):
                        if game.board[br][bc].winner == Player.NONE and not game.board[br][bc].is_full():
                            for sr in range(3):
                                for sc in range(3):
                                    if game.board[br][bc].board[sr][sc] == 0:
                                        moves.append((br*3 + sr, bc*3 + sc))
            else:
                br, bc = nb
                for sr in range(3):
                    for sc in range(3):
                         if game.board[br][bc].board[sr][sc] == 0:
                             moves.append((br*3 + sr, bc*3 + sc))

        if not moves:
            return self.evaluate(game)

        prev_next_coords = game.next_board_coords

        if maximizing:
            max_eval = -float('inf')
            for r, c in moves:
                game.make_move(r, c, Player.PLAYER)
                try:
                    eval_val = self.minimax(game, depth - 1, alpha, beta, False)
                finally:
                    game.undo_move(r, c, prev_next_coords)
                
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in moves:
                game.make_move(r, c, Player.OPPONENT)
                try:
                    eval_val = self.minimax(game, depth - 1, alpha, beta, True)
                finally:
                    game.undo_move(r, c, prev_next_coords)
                
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha: break
            return min_eval

    def get_best_move(self, game: UltimateBoard, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        self.start_time = time.time()
        best_move = valid_moves[0] if valid_moves else (0, 0)
        
        valid_moves.sort(key=lambda m: 0 if (m[0]%3==1 and m[1]%3==1) else 1)

        for depth in range(1, 10):
            try:
                current_best_move = None
                max_eval = -float('inf')
                alpha = -float('inf')
                beta = float('inf')
                
                prev_next = game.next_board_coords

                for r, c in valid_moves:
                    game.make_move(r, c, Player.PLAYER)
                    try:
                        eval_val = self.minimax(game, depth, alpha, beta, False)
                    finally:
                        game.undo_move(r, c, prev_next)

                    if eval_val > max_eval:
                        max_eval = eval_val
                        current_best_move = (r, c)
                    
                    alpha = max(alpha, eval_val)
                    
                    if time.time() - self.start_time > Config.TIME_LIMIT:
                        raise TimeoutError()
                
                if current_best_move:
                    best_move = current_best_move
            
            except TimeoutError:
                break
                
        return best_move

def game_loop() -> None:
    game_board = UltimateBoard()
    ai = AI()

    while True:
        try:
            line = input().split()
            if not line: break
            opponent_row, opponent_col = map(int, line)
            
            valid_action_count = int(input())
            valid_moves = []
            for _ in range(valid_action_count):
                valid_moves.append(tuple(map(int, input().split())))
            
            if opponent_row != -1:
                try:
                    game_board.make_move(opponent_row, opponent_col, Player.OPPONENT)
                except Exception:
                    pass

            move = ai.get_best_move(game_board, valid_moves)
            
            game_board.make_move(move[0], move[1], Player.PLAYER)

            print(f"{move[0]} {move[1]}")
            sys.stdout.flush()

        except EOFError:
            break

if __name__ == "__main__":
    game_loop()
