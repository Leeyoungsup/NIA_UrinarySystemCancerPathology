import numpy as np
import time
import datetime
import os
import warnings
import pandas as pd
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import warnings

import torchvision

import matplotlib.pyplot as plt
# 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore")


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


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


def Preprocessing(file_list, label_data):
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Preprocessing Start]')
    print(f'Preprocessing Start Time : {now_time}')
    frame_path = './frame/'
    for i in tqdm(range(len(file_list))):
        try:
            count = 0
            vidcap = torchvision.io.read_video(file_list[i])
            fps = int(vidcap[2]['video_fps'])
            video = np.array(vidcap[0], dtype=np.uint8)
            video_crop = np.zeros(
                (len(video)-1, video.shape[1], video.shape[2], 3))
            for j in range(len(video_crop)):
                video_crop[j] = video[j+1]-video[j]
            video_crop = video_crop.sum(axis=0)
            video_crop = video_crop.sum(axis=2)
            video_crop = ((video_crop/video_crop.max())*255).astype(np.uint8)
            y1 = np.where(video_crop > 100)[0].min()
            y2 = np.where(video_crop > 100)[0].max()
            x1 = np.where(video_crop > 100)[1].min()
            x2 = np.where(video_crop > 100)[1].max()
            video_name = os.path.basename(file_list[i])
            dst_label = label_data.loc[label_data["File Name"] == video_name]
            wake = str(dst_label['구분값'].item())
            Serial_Number = str(dst_label['일련번호'].item())
            file_name = Serial_Number+wake
            createDirectory(frame_path+file_name)
            for k in range(0, len(video), fps//5):
                img = Image.fromarray(video[k, y1:y2, x1:x2])
                im_new = expand2square(img, (0, 0, 0))
                im_new.resize((256, 256)).save(
                    frame_path+file_name+"/%06d.jpg" % count)
                count += 1
        except:
            print(file_list[i])
    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Preprocessing Time : {now_time}s Time taken : {end-start}')
    print(f'[Preprocessing End]')
