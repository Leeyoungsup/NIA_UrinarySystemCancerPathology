import os
import numpy as np
from glob import glob
from PIL import Image
import cv2
import slideio
import json
import time
import datetime
from skimage.draw import polygon2mask
from tqdm.auto import tqdm


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def size_ratio(scene, img_size):
    width = scene.rect[2]
    height = scene.rect[3]
    ratio = 0
    inverse_ratio = 0
    img_width = 0
    img_height = 0
    if width > height:
        ratio = img_size/width
        inverse_ratio = width/img_size
        img_width = img_size
        img_height = height*ratio
    else:
        ratio = img_size/height
        inverse_ratio = height/img_size
        img_height = img_size
        img_width = width*ratio

    return int(img_width), int(img_height), inverse_ratio


def Preprocessing(whi_list, image_path, normal_mask_path, tumor_mask_path):
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Preprocessing Start]')
    print(f'Preprocessing Start Time : {now_time}')
    json_list = [f.replace('/WSI/', '/json/') for f in whi_list]
    json_list = [f.replace('.tiff', '.json') for f in json_list]
    createDirectory(image_path)
    createDirectory(normal_mask_path)
    createDirectory(tumor_mask_path)
    for i in tqdm(range(len(whi_list))):
        slide = slideio.open_slide(whi_list[i], "GDAL")
        fileName = os.path.basename(os.path.splitext(whi_list[i])[0])
        num_scenes = slide.num_scenes
        scene = slide.get_scene(0)
        img_width, img_height, ratio = size_ratio(scene, 2048)
        svsWidth = scene.rect[2]
        svsHeight = scene.rect[3]
        slide_block = scene.read_block(
            (0, 0, svsWidth, svsHeight), size=(int(img_width), int(img_height)))
        image = cv2.cvtColor(slide_block, cv2.COLOR_BGR2RGB)
        tumor_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        normal_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        total_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        with open(json_list[i]) as f:
            json_object = json.load(f)
        polygon_count = len(json_object['files'][0]['objects'])
        image_shape = (img_height, img_width)
        for j in range(polygon_count):
            if json_object['files'][0]['objects'][j]['label'] == 'TP_tumor':
                polygon = np.array(
                    json_object['files'][0]['objects'][j]['coordinate'])*1/ratio
                polygon1 = np.copy(polygon)
                polygon1[:, 0] = polygon[:, 1]
                polygon1[:, 1] = polygon[:, 0]
                mask = polygon2mask(image_shape, polygon1)
                tumor_mask = mask+tumor_mask
        tumor_mask = np.where(tumor_mask > 0, 255, 0)
        tumor_mask = cv2.cvtColor(
            tumor_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        t, t_otsu = cv2.threshold(
            image[:, :, 1], -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        t_otsu = cv2.morphologyEx(t_otsu, cv2.MORPH_OPEN, k)
        total_mask = 255 - \
            cv2.cvtColor(t_otsu.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        normal_mask = np.where((total_mask-tumor_mask) > 200, 255, 0)

        cv2.imwrite(tumor_mask_path+fileName+'.tiff', tumor_mask)
        cv2.imwrite(normal_mask_path+fileName+'.tiff',
                    normal_mask.astype(np.uint8))
        cv2.imwrite(image_path+fileName+'.tiff', image)
    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Preprocessing Time : {now_time}s Time taken : {end-start}')
    print(f'[Preprocessing End]')
