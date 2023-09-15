from dataset import train_loader, test_loader
from model import BersonNetwork
from model_resnet import resnet34

import time
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = BersonNetwork().to(device)
# print(model)
# model = resnet34().to(device)
model = torch.load('./model_save/model_resnet3.pth', map_location=torch.device(device))
model.train()


learning_rate = 1e-3 #@param
batch_size = 64 #@param
epochs = 20 #@param
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f'):
      self.name = name
      self.fmt = fmt
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

  def __str__(self):
      fmtstr = '{name} {avg' + self.fmt + '}'
      return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')

start = time.time()
for i in range(epochs):
    for batch, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)
        data_time.update(time.time() - start)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(pred, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(acc1[0], X.size(0))
        top5.update(acc5[0], X.size(0))
    batch_time.update(time.time() - start)
    start = time.time()
    print(f"Epoch:{i + 1}: {batch_time}, {losses}, {top1}, {top5}")

torch.save(model, './model_save/model4.pth')