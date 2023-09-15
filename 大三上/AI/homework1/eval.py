from dataset import test_loader, transform_val, train_loader

import torch
from PIL import Image

LABELS = [
    "ape", "bear", "bison", "cat", 
    "chicken", "cow", "deer", "dog",
    "dolphin", "duck", "eagle", "fish", 
    "horse", "lion", "lobster", "pig", 
    "rabbit", "shark", "snake", "spider", 
    "turkey", "wolf"
]
LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat", 
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    8: "dolphin", 9: "duck", 10: "eagle", 11: "fish", 
    12: "horse", 13: "lion", 14: "lobster", 
    15: "pig", 16: "rabbit", 17: "shark", 18: "snake", 
    19: "spider", 20:  "turkey", 21: "wolf"
}
NUM_CLASSES = 22

model = torch.load("./model_save/model_shuffle_resnet3.pth", map_location=torch.device('cpu'))
# model = torch.load("./model_save/model_resnet3.pth")
model.eval()
labels = []
"""
image = Image.open("./Animals Dataset/train/cat/20.png").convert("RGB")
image = Image.open("./Animals Dataset/test/1.png").convert("RGB")
image = transform_val(image)
image = torch.unsqueeze(image, dim=0)
with torch.no_grad():
    y = model(image)
    label = torch.argmax(y)
    print(LABEL_MAP[label.item()])
"""
with torch.no_grad():
    for _, (images, _) in enumerate(test_loader):
        # y = model(images)
        y = model(images)
        batch_labels = torch.argmax(y, dim=1)
        labels.append(batch_labels)
ans = torch.cat(labels, 0).cpu().numpy()
print(ans)
print([LABEL_MAP[i] for i in ans])
