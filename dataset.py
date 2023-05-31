import cv2
import torch
import albumentations as A
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self, img_path, target_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.target_path = target_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.target_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        name = self.X[idx] + '.png'

        if self.transform is not None:
            augmentated = self.transform(image=img, mask=target)
            # np array => PIL image
            img = augmentated['image']
            target = augmentated['mask']

        img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        target = torch.from_numpy(target).long()

        return img, target, name


class TestDataset(Dataset):

    def __init__(self, img_path, target_path, X, transform=None):
        self.img_path = img_path
        self.target_path = target_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.target_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=target)
            img = aug['image']
            target = aug['mask']

        img = Image.fromarray(img)

        target = torch.from_numpy(target).long()

        return img, target
