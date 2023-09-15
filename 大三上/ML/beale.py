import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


# function
beale = lambda x1, x2: (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2

# basic data
w1_min, w1_max, w1_step = -4.5, 4.5, .2
w2_min, w2_max, w2_step = -4.5, 4.5, .2

minima = [3, 0.5]

w1, w2 = np.meshgrid(np.arange(w1_min, w1_max + w1_step, w1_step), np.arange(w2_min, w2_max + w2_step, w2_step))
losses = beale(w1, w2)


class Net(torch.nn.Module):
    def __init__(self, x1, x2) -> None:
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.tensor([x1]))
        self.x2 = torch.nn.Parameter(torch.tensor([x2]))
    def forward(self):
        return beale(self.x1, self.x2)


def optim(x1_start, x2_start, optimizer: str, lr, epochs):
    net = Net(x1_start, x2_start)
    optim = 0
    if optimizer == "sgd":
        optim = torch.optim.SGD(net.parameters(), lr)
    elif optimizer == "momentum":
        optim = torch.optim.SGD(net.parameters(), lr, momentum=0.9)
    elif optimizer == "adam":
        optim = torch.optim.Adam(net.parameters(), lr)
    else:
        print("invalid inputs")
        return
    
    x1_track, x2_track = [], []
    
    for i in range(epochs):
        optim.zero_grad()
        net().backward()

        x1_track.append(net.x1.item())
        x2_track.append(net.x2.item())

        optim.step()
    
    trajectory = np.array([x1_track, x2_track])
    return trajectory

epochs = 20000
optims = ['sgd', 'momentum', 'adam']
lrs = [0.001, 0.001, 0.1]

for i in range(3):
    t = optim(2.5, 2., optims[i], lrs[i], epochs)
    print("optims: {}; [{}, {}]".format(optims[i], t[0][epochs - 1], t[1][epochs - 1]))
    


"""
fig, ax = plt.subplots(figsize=(10, 6))
ax.contour(w1, w2, losses, levels=np.logspace(0, 5, 35), alpha=0.8, cmap=cm.jet)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
"""


