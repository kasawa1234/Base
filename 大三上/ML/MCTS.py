# using MCTS for tic-tac-toe
# N x N board, first o second x
import math
import numpy as np
import operator
from random import choice
import matplotlib.pyplot as plt

# origin status: one root node
# start from root node and 
# select: all way to child node, unless it still can be expanded
#   * during this process, using UCB to choose child node
# expand: expand unless game end
#   * if end, skip
# playout: play till end
# backpropagation: return value all the way up to root node

class Node:
    # define Node as each tree node
    def __init__(self, status, ava, depth, parent_index, index, side):
        super().__init__()
        self.qx = 0     # winning times for x
        self.qo = 0     # winning times for o
        self.n = 0      # visited times
        self.depth = depth      # times, 9 times means over
        self.childs_index = []      # record index of child nodes
        self.parent_index = parent_index      # record index of parent node, -1 is for root
        self.index = index      # index of itself
        self.move_ava = ava     # available moves remained
        self.move_ava_copy = ava.copy()
        self.side = side        # 1 for o and -1 for x
        self.status = status


def cal_UCB(qi, ni, N, C=2):
    return float(qi) / float(ni) + C * math.sqrt(math.log(N) / float(ni))

def win_judge(status, side):
    l = np.array([side, side, side])
    for i in range(3):
        if (status[i] == l).all()  or (status[:, i] == l).all():      # row and col
            return 1
        if (np.diag(status) == l).all() or (np.diag(np.rot90(status)) == l).all():      # diag
            return 1
    return 0    # no win or lose yet

def UCB(node: Node):
    # input current node
    # return index of selected child
    tmp = []
    side = node.side
    for index in node.childs_index:
        child = tree[index]
        if side == 1:   # o
            tmp.append(cal_UCB(child.qo, child.n, node.n))
        else:   # x
            tmp.append(cal_UCB(child.qx, child.n, node.n))
    return node.childs_index[np.argmax(np.array(tmp))]

def select_expand(cur_index, tree: list):
    if tree[cur_index].depth == 9:      # arrive the end
        return cur_index
    if not len(tree[cur_index].move_ava) == 0:      # can be expanded
        # expand one node and initialize
        tree[cur_index].childs_index.append(len(tree))
        cur_node = tree[cur_index]
        coor = choice(cur_node.move_ava)     # random choose next step and expand
        tree[cur_index].move_ava.remove(coor)       # remove used coor

        expand_status = np.copy(cur_node.status)
        expand_status[coor[0]][coor[1]] = cur_node.side

        expand_ava = cur_node.move_ava_copy.copy()
        expand_ava.remove(coor)

        expand_node = Node(expand_status, expand_ava, cur_node.depth + 1, cur_node.index, 
                           len(tree), -cur_node.side)
        tree.append(expand_node)
        return expand_node.index
    return select_expand(UCB(tree[cur_index]), tree)

def playout(node: Node):
    # input: node for playout
    # return: 1 for o, -1 for x and 0 for draw
    steps = node.depth
    side = node.side
    empty = node.move_ava_copy.copy()
    status = node.status
    while steps < 9:
        side = -side
        coor = choice(empty)
        empty.remove(coor)
        status[coor[0]][coor[1]] = side
        
        if win_judge(status, side):
            return side
        else:
            steps += 1
    return 0

def back(index, o, x, tree):
    while not index == -1:
        tree[index].qo += o
        tree[index].qx += x
        tree[index].n += 1
        index = tree[index].parent_index

ava_root = []
for i in range(3):
    for j in range(3):
        ava_root.append([i, j])
root = Node(status=np.zeros([3, 3]), ava=ava_root, depth=0, parent_index=-1, index=0, side=1)
tree = [root]

rec = []
for i in range(10000):
    select_index = select_expand(0, tree)
    result = playout(tree[select_index])
    o, x = 0, 0
    if result == 1:
        o = 1
    elif result == -1:
        x = 1
    back(select_index, o, x, tree)
    
    rec.append(tree[0].qo / tree[0].n)

plt.plot(np.arange(10000), rec)
plt.show()
