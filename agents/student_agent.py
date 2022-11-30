# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from collections import deque
import sys


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
        self.move_directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

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
        # dummy return
        #input("Next move?")
        return self.minimax_decision(chess_board, my_pos, adv_pos, max_step)

    def possible_moves(self, chess_board, my_pos, adv_pos, max_step):  # returns a set

        q = deque()
        q.append((my_pos, 0))
        pos_moves = set()
        visited = set()
        while len(q) > 0:
            (x, y), depth = q.popleft()
            if depth > max_step:
                break

            if (x, y) not in visited and not (x, y) == adv_pos:
                # check and add up
                if not chess_board[x, y, self.dir_map["u"]] :  # check for a barrier and if the opponent is positioned in the desired square.
                    pos_moves.add(((x, y), self.dir_map["u"]))
                    q.append(((x - 1, y), depth + 1))

                # check and add right
                if not chess_board[x, y, self.dir_map["r"]] :
                    pos_moves.add(((x, y), self.dir_map["r"]))
                    q.append(((x, y + 1), depth + 1))

                # check and add down
                if not chess_board[x, y, self.dir_map["d"]] :
                    pos_moves.add(((x, y), self.dir_map["d"]))
                    q.append(((x + 1, y), depth + 1))

                # check and add left
                if not chess_board[x, y, self.dir_map["l"]] :
                    pos_moves.add(((x, y), self.dir_map["l"]))
                    q.append(((x, y - 1), depth + 1))
                # enqueue neighbors with added depth
                # can only enqueue of the neighbor is reachable (check chess_board[r, c, dir])
                # add pos to visited
                visited.add((x, y))
        return pos_moves

    # TODO: Minimax function (option for depth limited)
    def minimax_decision(self, chess_board, my_pos, adv_pos, max_step):
        max = float('-inf')
        max_move = None
        moves = self.possible_moves(chess_board, my_pos, adv_pos,max_step)
        i = 0
        for op in moves:
            i +=1
            print(f"Move {i} out of {len(moves)}")
            print(op)
            new_pos, _ = op  # by taking the move, the player moves
            # The subsequent move will be the min player's, so max_player = False
            value = self.minimax_val(self.transition(chess_board, op), new_pos, adv_pos, False, max_step, 5)
            print(f"Current move is {op} with value {value}")
            if value == 1:  # In this case we found a win so we can stop. There will not be a bigger value
                print("stop early")
                return op
            if value > max:
                max = value
                max_move = op
        print(f"The max_move is {max_move} with value {max}")
        return max_move


    def minimax_val(self, chess_board, my_pos, adv_pos, max_player, max_step, depth):
        """
        Calculates the minimax value of a given board
        Returns: Float or int
        If max's turn, returns max of min player's next turn. If min's turn, returns min of max player's next turn.
        ------
        chess_board : Array (mxmx4)
        my_pos : Tuple
        adv_pos : Tuple
        max_player : bool
            True if it is the max player's turn. False if min_player's turn
        """
        #print("minimax_val executing")
        #print(depth)
        if depth == 0:  # depth limit reached
            return 0
        over, my_score, adv_score = StudentAgent.check_endgame(chess_board, my_pos, adv_pos)

        # If leaf node
        if over: # meaning the game is over and thus it is a leaf node
            # calculate score. If negative, then min player won. If equal to 0, then it's a tie. Else max won
            # Returns 1 for win, -1 for loss, 0 for a tie
            score = my_score-adv_score
            if score > 0:
                return 1
            elif score < 0:
                return score
            else:
                return 0

        if max_player:
            value = float('-inf')
            for op in self.possible_moves(chess_board, my_pos, adv_pos, max_step) :
                new_board = self.transition(chess_board, op)
                # by taking the move, the max player moves
                new_pos, _ = op # need to calculate the new position
                value = max(value, self.minimax_val(new_board, new_pos, adv_pos, (not max_player), max_step, depth-1 ))
            return value

        else: # in the case of min player
            value = float('inf')
            for op in self.possible_moves(chess_board, adv_pos, my_pos, max_step):  # for every possible subsequent move
                new_board = self.transition(chess_board, op)  # child node
                # by taking the move, the min player moves
                new_pos, _ = op  # need to calculate the new position
                value = min(value, self.minimax_val(new_board, my_pos, new_pos, (not max_player), max_step, depth-1))
            return value

    @staticmethod
    def check_endgame(chess_board, my_pos, adv_pos):
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
        board_size = chess_board.shape[0]
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
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
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        else:
            return True, p0_score, p1_score

    def transition(self, board, move):

        # Move must be formatted as such
        (x, y), dir = move
        result = board.copy()
        # Set the barrier to True
        result[x, y, dir] = True
        # Set the opposite barrier to True
        move = self.move_directions[dir]
        result[x + move[0], y + move[1], self.opposites[dir]] = True
        return result


