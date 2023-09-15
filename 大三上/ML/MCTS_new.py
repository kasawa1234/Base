import numpy as np
import random
import copy
from collections import defaultdict

"""
INFO:
    Using 1 and -1 for both sides
"""

SIZE = 7
MAX_EPOCHS = 5000

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
        self.results[1], self.results[-1] = 0, 0

    def untried():

