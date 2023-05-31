# Util
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Troch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

# Computer Vision Lib
import albumentations as A

# Std
import time
import os

from UNet import UNet
from dataset import TrainDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set your dataset dir
IMAGE_PATH = '../cnu_senior_project/dataset/dataset/LPCVC_Train/IMG/train/'
MASK_PATH = '../cnu_senior_project/dataset/dataset/LPCVC_Train/GT/train/'
n_classes = 14
model_name = 'UNet'

# Training initialization
batch_size= 3
max_lr = 1e-3
epoch = 500
weight_decay = 3e-3
df = None
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

if model_name == 'UNet':
    model = UNet()
elif model_name == 'AUNet':
    model = None

test_size = 0.0001

file_names = []
for _, _, filenames in os.walk(IMAGE_PATH):
    for filename in filenames:
        file_names.append(filename.split('.')[0])

df = pd.DataFrame({'id': file_names}, index = np.arange(0, len(file_names)))

x_train, x_test = train_test_split(df['id'].values, test_size=test_size, random_state=19)
x_train, x_val = train_test_split(x_train, test_size=test_size, random_state=19)


transform_train = A.Compose([A.HorizontalFlip(), A.VerticalFlip()])
transform_val = A.Compose([A.HorizontalFlip(), A.VerticalFlip()])

train_set = TrainDataset(IMAGE_PATH, MASK_PATH, x_train, mean, std, transform_train, patch=False)
val_set = TrainDataset(IMAGE_PATH, MASK_PATH, x_val, mean, std, transform_val, patch=False)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

def miou(pred_target, target, smooth=1e-10, n_classes=14):
    with torch.no_grad():
        pred_target = F.softmax(pred_target, dim=1)
        pred_target = torch.argmax(pred_target, dim=1)
        pred_target = pred_target.contiguous().view(-1)
        target = target.contiguous().view(-1)

        iou_per_class = []
        for label in range(0, n_classes):
            true_class = pred_target == label
            true_label = target == label

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model,
        train_loader,
        criterion, optimizer, scheduler):
    torch.cuda.empty_cache()
    train_losses = []
    train_iou = []
    lrs = []

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        start_time = time.time()
        running_loss = 0
        iou_score = 0

        model.train()
        for idx, data in enumerate(tqdm(train_loader)):
            image_s, target_s, name = data

            image = image_s.to(device); mask = target_s.to(device);
            # Forward
            output = model(image)
            loss = criterion(output, mask)

            # Evaluation metrics
            iou_score += miou(output, mask)

            # Backward
            loss.backward()

            # Update weight
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()
            train_losses.append(running_loss/len(train_loader))

            train_iou.append(iou_score/len(train_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Train miou:{:.3f}..".format(iou_score/len(train_loader)),
                  "Time: {:.2f}m".format((time.time()-start_time)/60))

    history = {'train_loss' : train_losses, 'train_miou' :train_iou, 'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

history = fit(epoch, model, train_loader, criterion, optimizer, sched)
torch.save(model.state_dict(), f'{model_name}-{epoch}.pkl')

def plot_loss(history):
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('plot_loss.png')
    plt.close()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_miou', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('plot_score.png')
    plt.close()

plot_loss(history)
plot_score(history)
