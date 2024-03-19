import numpy as np

class DuoOthello:
    def __init__(self):
        self.token = ''
        self.rtime = 0.0
        self.op_rtime = 0.0
        self.state = np.empty((12, 12), dtype=str)
        self.blank_pos = '.'
        
        self.state_weights = np.array(
                               [[4, -3, 2, 2, 2, 2, 2, 2, 2, 2, -3, 4],
                                [-3, -4, -1, -1, -1, -1, -1, -1, -1, -1, -4, -3],
                                [2, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 2],
                                [2, -1, 0, 1, 0, 0, 0, 0, 1, 0, -1, 2],
                                [2, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 2],
                                [2, -1, 0, 0, 0, 1, 1, 0, 0, 0, -1, 2],
                                [2, -1, 0, 0, 0, 1, 1, 0, 0, 0, -1, 2],
                                [2, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 2],
                                [2, -1, 0, 1, 0, 0, 0, 0, 1, 0, -1, 2],
                                [2, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 2],
                                [-3, -4, -1, -1, -1, -1, -1, -1, -1, -1, -4, -3],
                                [4, -3, 2, 2, 2, 2, 2, 2, 2, 2, -3, 4]])

    def parse_input(self, file):
        with open(file, "r") as f:
            input_lines = f.readlines()
            self.token = input_lines[0].strip()
            self.rtime, self.op_rtime = map(float, input_lines[1].split())
            self.state = np.array([list(row.strip()) for row in input_lines[2:]])

    def eval_state(self, state, token):
        token_coins = np.sum(state == token)
        op_coins = np.sum(state == ('O' if token == 'X' else 'X'))
        disc_diff = token_coins - op_coins
        mv = 0
        r, c = np.where(state == token)
        for j in range(len(r)):
            mv += self.state_weights[r[j]][c[j]]
        
        stability = np.sum(np.multiply(state == token, self.state_weights))
        total_heuristic= 1 * disc_diff + 2 * stability + 1 * mv

        return total_heuristic

    def is_valid(self, state, token, move):
        r, c = move[0],move[1]

        adj = [(0, 1), (0, -1),(1, 0), (-1, 0), (1, 1), (-1, -1),(1, -1), (-1, 1)]
        
        if state[r, c] != self.blank_pos:
            return False

        for i, j in adj:
            diffi, diffj = r + i, c + j
            flag = False

            while 0 <= diffi < 12 and 0 <= diffj < 12:
                if state[diffi, diffj] == self.blank_pos:
                    break
                elif state[diffi, diffj] == token:
                    if flag:
                        return True
                    else:
                        break
                else:
                    flag = True

                diffi += i
                diffj += j

        return False

    def move(self, state, token, move):
        r, c = move
        state[r, c] = token

        adj = [(0, 1), (0, -1),(1, 0), (-1, 0), (1, 1), (-1, -1),(1, -1), (-1, 1)]

        for i, j in adj:
            diffi, diffj = r + i, c + j
            flip_op = []

            while 0 <= diffi < 12 and 0 <= diffj < 12:
                if state[diffi, diffj] == self.blank_pos:
                    break
                elif state[diffi, diffj] == token:
                    for a, b in flip_op:
                        state[a, b] = token
                    break
                else:
                    flip_op.append((diffi, diffj))

                diffi += i
                diffj += j

    def get_legal_pos(self, state, token):
        legal_pos = []

        for i in range(12):
            for j in range(12):
                if self.is_valid(state, token, (i, j)):
                    legal_pos.append((i, j))

        return legal_pos

    


    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_token):
        if depth == 0:
            return self.eval_state(state, self.token)

        legal_pos = self.get_legal_pos(state, self.token if maximizing_token else ('O' if self.token == 'X' else 'X'))

        if maximizing_token:
            value = float('-inf')
            for move in legal_pos:
                next_state = np.copy(state)
                self.move(next_state, self.token, move)
                value = max(value, self.alpha_beta_pruning(next_state, depth - 1, alpha, beta, not maximizing_token))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in legal_pos:
                next_state = np.copy(state)
                self.move(next_state, 'O' if self.token == 'X' else 'X', move)
                value = min(value, self.alpha_beta_pruning(next_state, depth - 1, alpha, beta, not maximizing_token))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def best_position(self, state):
        legal_pos = self.get_legal_pos(state, self.token)
        op_pos = None
        opt_value = float('-inf')
        for move in legal_pos:
            next_state = np.copy(state)
            self.move(next_state, self.token, move)
            value = self.alpha_beta_pruning(next_state, 3, float('-inf'), float('inf'), True)
            if op_pos is None or (self.token == 'X' and value > opt_value) or (self.token == 'O' and value > opt_value):
                opt_value = value
                op_pos = move

        return op_pos

    def write_index_to_position(self, pos):
        alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l']
        row = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        square = alphabet[pos[1]] + str(row[pos[0]])
        with open("output.txt", "w") as f:
            f.write(square)


if __name__ == "__main__":
    do = DuoOthello()
    do.parse_input("input.txt")
    best_position = do.best_position(do.state)
    #print("Best Move:", op_pos)
    do.write_index_to_position(best_position)