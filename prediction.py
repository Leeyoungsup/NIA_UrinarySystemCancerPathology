import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch
from torchinfo import summary
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.functional as F
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import time
import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
image_count = 50
img_size = 512
tf = ToTensor()


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class CustomDataset(Dataset):
    def __init__(self, image_list, label_list, file_list):
        self.img_path = image_list
        self.label = label_list
        self.file_list = file_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_tensor = self.img_path[idx]
        label_tensor = self.label[idx]
        path = os.path.splitext(os.path.basename(self.file_list[idx]))[0]
        return image_tensor, label_tensor, path


def data_load(image_list):
    tumor_mask_list = [f.replace('/image/', '/polygon/TP_tumor/')
                       for f in image_list]
    normal_mask_list = [
        f.replace('/image/', '/polygon/NT_normal/') for f in image_list]

    image = torch.zeros((len(image_list), 3, img_size, img_size))
    mask = torch.zeros((len(image_list), 3, img_size, img_size))

    for i in tqdm(range(len(image_list))):
        img = 1 - \
            tf(np.array(expand2square(Image.open(
                image_list[i]), (255, 255, 255)).resize((img_size, img_size))))
        msk_tumor = np.array((expand2square(Image.open(
            tumor_mask_list[i]), (0, 0, 0)).convert('L')).resize((img_size, img_size)))
        msk_normal = np.array((expand2square(Image.open(
            normal_mask_list[i]), (0, 0, 0)).convert('L')).resize((img_size, img_size)))
        msk_back = np.where((msk_tumor+msk_normal) == 0, 255, 0)
        image[i] = img
        mask[i, 0] = tf(msk_back)
        mask[i, 1] = tf(msk_tumor)
        mask[i, 2] = tf(msk_normal)

    dataset = CustomDataset(image, mask, image_list)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    return dataloader


def dice_loss(pred, target, num_classes=3):
    smooth = 1.
    dice_per_class = torch.zeros(num_classes).to(pred.device)

    for class_id in range(num_classes):
        pred_class = pred[:, class_id, ...]
        target_class = target[:, class_id, ...]

        intersection = torch.sum(pred_class * target_class)
        A_sum = torch.sum(pred_class * pred_class)
        B_sum = torch.sum(target_class * target_class)

        dice_per_class[class_id] = 1 - \
            (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return dice_per_class


def Predict(image_list):
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Predict Start]')
    print(f'Predict Start Time : {now_time}')
    print('Data load...')
    dataloader = data_load(image_list)
    model = smp.UnetPlusPlus(
        'efficientnet-b6', in_channels=3, classes=3).to(device)
    model.load_state_dict(torch.load('../../model/Best_model.pt'))
    total_path = []
    print('predict...')
    total_prob = torch.zeros(
        (len(dataloader), 3, img_size, img_size)).to(device)
    total_y = torch.zeros((len(dataloader), 3, img_size, img_size)).to(device)
    total_dice = torch.zeros((len(dataloader), 3)).to(device)
    model.eval()
    count = 0
    val_running_loss = 0.0
    acc_loss = 0
    test = tqdm(dataloader)

    with torch.no_grad():
        for x, y, path in test:
            y = y.to(device).float()
            x = x.to(device).float()
            predict = model(x).to(device)
            cost = torch.mean(dice_loss(predict, y, num_classes=3))  # cost 구함
            acc = 1-torch.mean(dice_loss(predict, y, num_classes=3))
            val_running_loss += cost.item()
            acc_loss += acc
            prob_pred = predict
            total_path.append(path)
            total_prob[count] = predict.squeeze(dim=1)
            total_y[count] = y.squeeze(dim=1)
            total_dice[count] = dice_loss(predict, y, num_classes=3)
            count += 1
    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Predict Time : {now_time}s Time taken : {end-start}')
    print(f'[Predict End]')
    return total_path, total_y.cpu(), total_prob.cpu(), total_dice.cpu()
