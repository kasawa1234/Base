import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NUM_CLASSES = 22
class BersonNetwork(nn.Module):
  def __init__(self):
    super(BersonNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.c1 = nn.Conv2d(3, 20, 5, 2, 0) # (111 * 111) * 20
    self.c2 = nn.Conv2d(20, 1, 5, 1, 0) # (107 * 107) * 1
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(11449, 128),
        #!Tips: You should calculate the number of neurons.
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, NUM_CLASSES),
    )

  def forward(self, x):
    x = self.c1(x)
    x = F.relu(x) #!Question: What's the difference between torch.nn.relu() and torch.nn.F.relu()
    x = self.c2(x)
    x = F.relu(x)
    x = x.view(x.size(0), -1)
    logits = self.linear_relu_stack(x)
    return logits
