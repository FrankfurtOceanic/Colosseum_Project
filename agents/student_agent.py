# Student agent: Add your own agent here
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
from collections import deque, defaultdict
from datetime import timedelta, datetime
from math import log, sqrt
from random import choice
from typing import Union
import sys

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP_MAP = {0: 2, 1: 3, 2: 0, 3: 1}
DIR_MAP = {"u": 0, "r": 1, "d": 2, "l": 3}

def check_game_over(board, pos: tuple, adv: tuple) -> tuple[bool, int, int]:
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

    def union(this, other):
        father[this] = other

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                MOVES[1:3]
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


def transition(board, move: tuple[tuple[int, int], int]):
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
    opp_move = MOVES[dir] # opposite move
    result[x + opp_move[0], y + opp_move[1], OPP_MAP[dir]] = True
    return result


def get_moves(board, pos: tuple[int, int], adv: tuple[int, int], max_step: int) -> list[tuple[tuple[int, int], int]]:
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
            if not board[x, y, DIR_MAP["u"]] :  # check for a barrier and if the opponent is positioned in the desired square.
                moves.add(((x, y), DIR_MAP["u"]))
                q.append(((x - 1, y), depth + 1))

            # check and add right
            if not board[x, y, DIR_MAP["r"]] :
                moves.add(((x, y), DIR_MAP["r"]))
                q.append(((x, y + 1), depth + 1))

            # check and add down
            if not board[x, y, DIR_MAP["d"]] :
                moves.add(((x, y), DIR_MAP["d"]))
                q.append(((x + 1, y), depth + 1))

            # check and add left
            if not board[x, y, DIR_MAP["l"]] :
                moves.add(((x, y), DIR_MAP["l"]))
                q.append(((x, y - 1), depth + 1))
            # enqueue neighbors with added depth
            # can only enqueue of the neighbor is reachable (check chess_board[r, c, dir])
            # add pos to visited
            visited.add((x, y))
    return list(moves)

class MonteCarloNode:
    player: bool
    depth: int
    move: tuple[tuple[int, int], int]
    
    def __init__(self, player: bool, depth: int, move: tuple[tuple[int, int], int]) -> None:
        self.player = player
        self.depth = depth
        self.move = move

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MonteCarloNode):
            return (self.player == other.player) and (self.depth == other.depth) and (self.move == other.move)
        return False
    
    def __hash__(self) -> int:
        return hash((self.player, (self.depth, self.move)))

DEFAULT_DEPTH_LIMIT = int(50)
DEFAULT_SEARCH_TIME = 0.5
SETUP_TIME = 3.0
SETUP_DEPTH_LIMIT = int(100)
TOL_TIME = 0.1

class MonteCarloTreeSearcher:
    max_step: int
    depth: int
    playouts: dict[MonteCarloNode, int]
    wins: dict[MonteCarloNode, int]
    weight: float

    def __init__(self, max_step: int, **kwargs) -> None:
        self.max_step = max_step
        self.depth = 0
        self.playouts = defaultdict(int)
        self.wins = defaultdict(int)
        self.weight = kwargs.get("weight", 1.4)
    
    def expand(self, node: MonteCarloNode) -> None:
        self.wins[node] = 0
        self.playouts[node] = 0

    def expanded(self, node: MonteCarloNode) -> bool:
        return (node in self.playouts)

    def total(self, nodes: list[MonteCarloNode]) -> int:
        return sum(self.playouts.get(node) for node in nodes)
    
    def uct(self, node: MonteCarloNode, log_total: float) -> float:
        return (self.wins[node]/self.playouts[node]) + self.weight * sqrt(log_total/self.playouts[node])

    def win_rate(self, node: MonteCarloNode) -> float:
        return self.wins.get(node, 0)/self.playouts.get(node, 1)

    def select(self, nodes: list[MonteCarloNode]) -> MonteCarloNode:
        if all(self.expanded(node) for node in nodes):
            log_total = log(self.total(nodes))
            node = max(nodes, key=lambda n: self.uct(n, log_total))
        else:
            node = choice(nodes)
        return node

    def update(self, nodes: set[MonteCarloNode], winner: Union[None, bool]) -> None:
        for node in nodes:
            if node not in self.playouts:
                continue
            player = node.player   
            self.playouts[node] += 1
            if winner is not None:
                if winner == player:
                    self.wins[node] += 1

    def simulate(self, board, pos: tuple[int, int], adv: tuple[int, int], **kwargs) -> None:
        visited = set()
        player = True
        winner = None
        max_step = self.max_step

        depth_limit = kwargs.get("depth_limit", DEFAULT_DEPTH_LIMIT)
        expand = True
        for depth in range(self.depth, self.depth + depth_limit + 1):
            # Get all the possible next moves.
            moves = get_moves(board, pos, adv, max_step)

            # Associate to each move a node in the tree
            nodes = [MonteCarloNode(player, depth, move) for move in moves]

            # Select the next node.
            node = self.select(nodes)

            # Get the node move.
            move = node.move

            # Apply the move to the board.
            board = transition(board, move)

            # Expand the node if necessary.
            if expand and not self.expanded(node):
                expand = False
                self.expand(node)

            # Mark node as visited.
            visited.add(node)

            # Check if the node is a terminal.
            game_over, score, adv_score = check_game_over(board, pos, adv)
            if game_over:
                if score > adv_score:
                    winner = player
                else:
                    winner = not player 
                break

            # Go to the other player.
            player = not player

            # Update positions of players.
            pos = adv
            adv, _ = move

        # Update all the visited nodes.
        self.update(visited, winner)

    def search(self, board, pos: tuple[int, int], adv: tuple[int, int], **kwargs) -> None:
        max_simulations = kwargs.get("max_simulations")
        seconds = kwargs.get("search_time", DEFAULT_SEARCH_TIME) - TOL_TIME
        search_time = timedelta(seconds=seconds)

        # Simulate games until the maximum number
        # of simulations is reached or the time limit
        # is reached.
        num_simulations = 0
        start = datetime.utcnow()
        while datetime.utcnow() - start < search_time and num_simulations < max_simulations:
            self.simulate(deepcopy(board), deepcopy(pos), deepcopy(adv), **kwargs)
            num_simulations += 1

    def choose(self, board, pos: tuple[int, int], adv: tuple[int, int], **kwargs) -> tuple[tuple[int, int], int]:
        # Increase the depth for the next move.
        self.depth += 1

        # Search in the tree.
        self.search(deepcopy(board), deepcopy(pos), deepcopy(adv), **kwargs)

        # Get the list of possible moves.
        moves = get_moves(board, pos, adv, self.max_step)

        # Choose the move with maximum estimated win rate.
        chosen = max(moves, key=lambda m: self.win_rate(MonteCarloNode(True, self.depth, m)))
        
        # Increase the depth for the opponent move.
        self.depth += 1

        # Return the chosen move.
        return chosen 

class AgentSearcher:
    def __init__(self) -> None:
        self.mcts = None
        self.max_simulations = 1

    def move(self, board, pos: tuple[int, int], adv: tuple[int, int], max_step: int) -> tuple[tuple[int, int], int]:
        kwargs = {}
        if self.mcts is None:
            # Initialize the monte carlo tree searcher.
            # Put the search time as the setup time.
            # Put the depth limit as the setup depth limit.
            # Compute the maximum number of simulations.
            self.mcts = MonteCarloTreeSearcher(max_step)
            kwargs["search_time"] = SETUP_TIME
            kwargs["depth_limit"] = SETUP_DEPTH_LIMIT
            self.max_simulations = board.shape[0] * max_step

        kwargs["max_simulations"] = self.max_simulations
        return self.mcts.choose(board, pos, adv, **kwargs)

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

        # Enable autoplay
        self.autoplay = True

        # Initialize the agent searcher.
        self.searcher = AgentSearcher()

    def step(self, board, pos: tuple[int, int], adv: tuple[int, int], max_step: int) -> tuple[tuple[int, int], int]:
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - board: a numpy array of shape (x_max, y_max, 4)
        - pos: a tuple of (x, y)
        - adv: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        return self.searcher.move(board, pos, adv, max_step)