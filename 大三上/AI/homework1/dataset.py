import os

from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset
from torch.utils.data import DataLoader
from PIL import Image

data_path = './Animals Dataset'
batch_size = 16
num_workers = 0

transform_labeled = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(size=[224, 224]),     # origin 225*225, resnet 224*224
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])      # for labeled data to train
transform_val = transforms.Compose([
    transforms.Resize(size=[224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])      # for validate and test

train_path = (os.path.join(data_path, 'train'))
test_path = (os.path.join(data_path, 'test'))

train_dataset = ImageFolder(
    train_path,
    transform_labeled
)
class MyDataset(VisionDataset):
    def __getitem__(self, index):
        img = self.transform(Image.open(self.root + "/{}.png".format(index)).convert('RGB'))
        return img, index
    def __len__(self):
        return len(os.listdir(self.root))

test_dataset = MyDataset(test_path, transform=transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False
)
