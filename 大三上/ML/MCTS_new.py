import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

"""
INFO:
    Using 1 for first, -1 for second
    Using UCB
"""

SIZE = 7
MAX_EPOCHS = 5000
C = 0.5

class Node():
    def __init__(self, state, parent=None, parent_action=None) -> None:
        super().__init__()
        self.state = state      # state of this node
        self.parent = parent    # parent node
        self.parent_action = parent_action

        self.childs = []
        self.remain_choices = None
        self.remain_choices = self.untried()

        self.num_visited = 0
        self.results = defaultdict(int)     # use it easier
        self.results[1], self.results[-1] = 0, 0        # record both sides at once

    def w(self):
        return self.results[self.state.order]

    def w_opp(self):
        return self.results[-self.state.order]
    
    def n(self):
        return self.num_visited
    
    def expand(self):
        action = self.remain_choices.pop()
        next_state = self.state.play(action)
        child_node = Node(next_state, parent=self, parent_action=action)     # create new child
        self.childs.append(child_node)      # add new child in cur node
        return child_node
    
    def untried(self):
        return copy.deepcopy(self.state.get_legal_actions())
    
    def best_child(self):
        w = [(c.w() / c.n()) + C * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.childs]
        return self.childs[np.argmax(w)]

    def random_child(self):
        return random.choice(self.childs)
    
    def is_terminal(self):
        return self.state.is_game_over()
    
    def is_fully_expand(self):
        return len(self.remain_choices) == 0
    
    def select(self):
        cur = self
        # if not terminal and fully expanded, expand one and return it
        # if is terminal, end
        # else, choose one child node and recurse
        while not cur.is_terminal():
            if not cur.is_fully_expand():
                return cur.expand()
            else:
                cur = cur.best_child()
        return cur
    
    def rollout(self):      # rollout didn't change Node, so using state instead
        cur = self.state
        while not cur.is_game_over():
            possible_moves = cur.get_legal_actions()
            action = possible_moves[np.random.randint(len(possible_moves))]
            cur = cur.play(action)
        return cur.game_result()        # return who win (0 for no one)
    
    def back(self, result):
        self.num_visited += 1
        self.results[result] += 1
        if self.parent:
            self.parent.back(result)


class State():
    def __init__(self, init_state, order) -> None:
        self.state = init_state
        self.order = order
    
    def play(self, action):
        board = copy.deepcopy(self.state)
        board[action[0]][action[1]] = self.order
        return State(board, -self.order)
    
    def is_game_over(self):
        return not (self.game_result() == 2)
    
    def game_result(self):
        # row
        for i in range(2, SIZE - 2):
            for j in range(SIZE):
                if not self.state[i][j] == 0:
                    if self.state[i - 2][j] == self.state[i - 1][j] == self.state[i][j] == self.state[i + 1][j] == self.state[i + 2][j]:
                        return self.state[i][j]
        for i in range(SIZE):
            for j in range(2, SIZE - 2):
                if not self.state[i][j] == 0:
                    if self.state[i][j - 2] == self.state[i][j - 1] == self.state[i][j] == self.state[i][j + 1] == self.state[i][j + 2]:
                        return self.state[i][j]
        for i in range(2, SIZE - 2):
            for j in range(2, SIZE - 2):
                if not self.state[i][j] == 0:
                    if self.state[i - 2][j - 2] == self.state[i - 1][j - 1] == self.state[i][j] == self.state[i + 1][j + 1] == self.state[i + 2][j + 2]:
                        return self.state[i][j]
        
        flag = 0
        for i in range(SIZE):
            for j in range(SIZE):
                if self.state[i][j] == 0:
                    flag = 1
        if flag == 0:
            return 0        # no empty
        return 2
    
    def get_legal_actions(self):
        legal_actions = []
        for i in range(SIZE):
            for j in range(SIZE):
                if self.state[i][j] == 0:
                    legal_actions.append([i, j])
        return legal_actions


if __name__ == '__main__':
    rec = []
    state = State(np.zeros([SIZE, SIZE]), 1)
    root = Node(state)
    
    for i in range(MAX_EPOCHS):
        selected = root.select()
        result = selected.rollout()
        selected.back(result)

        rec.append(float(root.w()) / float(root.n()))
        
        if (i + 1) % 100 == 0:
            print(root.w(), root.w_opp(), root.n())

plt.plot(np.arange(len(rec)), rec)
plt.show()
