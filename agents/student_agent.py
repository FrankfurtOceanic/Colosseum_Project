# Student agent: Add your own agent here
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
from collections import deque, defaultdict
from datetime import timedelta, datetime
from math import log, sqrt
from random import choice
import sys

def check_game_over(board, pos, adv, moves_arr=((-1, 0), (0, 1), (1, 0), (0, -1))):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    board_size = board.shape[0]
    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                moves_arr[1:3]
            ):  # Only check down and right
                if board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(pos))
    p1_r = find(tuple(adv))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    else:
        return True, p0_score, p1_score


def transition(board, move, moves_arr=((-1, 0), (0, 1), (1, 0), (0, -1)), op_map={0: 2, 1: 3, 2: 0, 3: 1}):
    """
    Transitions the input board with the move. Does not verify the validity of the move
    Parameters
    ----------
    board: input board
    move: the desired move

    Returns
    -------
    Returns a new board with the barrier placed (does not modify the input board)
    """
    # Move must be formatted as such
    (x, y), dir = move
    result = deepcopy(board)

    # Set the barrier to True
    result[x, y, dir] = True

    # Set the opposite barrier to True
    op_move = moves_arr[dir] # opposite move
    result[x + op_move[0], y + op_move[1], op_map[dir]] = True
    return result


def next_moves(board, pos, adv, max_step, dir_map={"u": 0, "r": 1, "d": 2, "l": 3}):
    """
    Calculates all possible moves for a given position, board, opponent position, and step size
    Parameters
    ----------
    board
    pos: tuple
    the player's position in a tuple (x,y)
    adv: tuple
    the opponent's position in a tuple (x,y)
    max_step: int
    the maximum number of step the player can take
    dir_map:

    Returns
    -------
    set
    returns a set of tuples. Each tuple is a possible move in the format ((x, y), dir)
    """
    q = deque()
    q.append((pos, 0))
    moves = set()
    visited = set()
    while len(q) > 0:
        (x, y), depth = q.popleft()
        if depth > max_step:
            break

        if (x, y) not in visited and not (x, y) == adv:
            # check and add up
            if not board[x, y, dir_map["u"]] :  # check for a barrier and if the opponent is positioned in the desired square.
                moves.add(((x, y), dir_map["u"]))
                q.append(((x - 1, y), depth + 1))

            # check and add right
            if not board[x, y, dir_map["r"]] :
                moves.add(((x, y), dir_map["r"]))
                q.append(((x, y + 1), depth + 1))

            # check and add down
            if not board[x, y, dir_map["d"]] :
                moves.add(((x, y), dir_map["d"]))
                q.append(((x + 1, y), depth + 1))

            # check and add left
            if not board[x, y, dir_map["l"]] :
                moves.add(((x, y), dir_map["l"]))
                q.append(((x, y - 1), depth + 1))
            # enqueue neighbors with added depth
            # can only enqueue of the neighbor is reachable (check chess_board[r, c, dir])
            # add pos to visited
            visited.add((x, y))
    return list(moves)

class MonteCarloTreeSearcher:
    def __init__(self, max_step: int, **kwargs) -> None:
        self.max_step = max_step

        seconds = kwargs.get("search_time", 1.90)
        self.search_time = timedelta(seconds=seconds)

        self.max_search_depth = kwargs.get("max_search_depth", 100)
        self.exploration_weight = kwargs.get("expoloration_weight", 1.4)

        self.depth = 0
        self.Q = defaultdict(int)
        self.N = defaultdict(int)

    def state(self, player, depth, move):
        return (player, (move, depth))

    def win_rate(self, move, depth):
        state = self.state(True, depth, move)
        return self.Q.get(state, 0)/self.N.get(state, 1)
    
    def expand(self, player, depth, move):
        state = self.state(player, depth, move)
        self.N[state] = 0
        self.Q[state] = 0

    def expanded(self, player, depth, move):
        return self.state(player, depth, move) in self.N

    def all_expanded(self, player, depth, moves):
        return all(self.expanded(player, depth, move) for move in moves)
    
    def total(self, player, depth, moves):
        return sum(self.N.get(self.state(player, depth, move)) for move in moves)
    
    def uct(self, player, depth, move, log_total):
        state = self.state(player, depth, move)
        return (self.Q[state]/self.N[state]) + self.exploration_weight * sqrt(log_total/self.N[state])

    def choose(self, board, pos, adv):
        self.depth += 1
        depth = self.depth
        moves = next_moves(board, pos, adv, self.max_step)    
        self.board = board
        self.pos = pos
        self.adv = adv

        self.simulate()

        best_move = moves[0]
        best_win_rate = self.win_rate(best_move, depth)

        for move in moves[1:]:
            win_rate = self.win_rate(move, depth)
            if win_rate > best_win_rate:
                best_move = move
                best_win_rate = win_rate

        self.depth += 1
        return best_move

    def simulate(self):
        visited = set()
        player = True
        winner = None
        depth = self.depth
        max_step = self.max_step
        board = deepcopy(self.board)
        pos = deepcopy(self.pos)
        adv = deepcopy(self.adv)

        expand = True
        for _ in range(self.max_search_depth):
            moves = next_moves(board, pos, adv, max_step)
            if self.all_expanded(player, depth, moves):
                log_total = log(self.total(player, depth, moves))
                next_move = moves[0]
                best_uct = self.uct(player, depth, next_move, log_total)
                for move in moves[1:]:
                    uct = self.uct(player, depth, move, log_total)
                    if uct > best_uct:
                        next_move = move
                        best_uct = uct
            else:
                next_move = choice(moves)

            move = next_move
            board = transition(board, move)

            if expand and not self.expanded(player, depth, move):
                expand = False
                self.expand(player, depth, move)
            
            visited.add(self.state(player, depth, move))

            game_over, score, adv_score = check_game_over(board, pos, adv)

            if game_over:
                if score > adv_score:
                    winner = player
                else:
                    winner = not player 
                break

            player = not player
            pos = adv
            (x, y), _ = move
            adv = (x, y)
            depth += 1
        
        for state in visited:
            player, (move, depth) = state
            if state not in self.N:
                continue
            self.N[state] += 1
            if winner is not None:
                if winner == player:
                    self.Q[state] += 1
class AgentSearcher:
    def __init__(self) -> None:
        self.mcts = None
    
    def move(self, board, pos, adv, max_step):
        if self.mcts is None:
            self.mcts = MonteCarloTreeSearcher(max_step)
        
        return self.mcts.choose(board, pos, adv)

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # Enable autoplay
        self.autoplay = True

        # Initialize the agent searcher.
        self.searcher = AgentSearcher()

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        move = self.searcher.move(chess_board, my_pos, adv_pos, max_step)
        return move