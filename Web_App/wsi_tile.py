import os
import sys
import numpy as np 
# import SlideRunner.general.dependencies
# from SlideRunner.dataAccess.database import Database
# from SlideRunner.dataAccess.annotations import *
import openslide
# import sqlite3
# import cv2
# from pathlib import Path
import pandas as pd
# import random
import matplotlib.pyplot as plt
from openslide import open_slide
# from os import system
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageDraw
import torch


def get_center_of_tiles(col, row, patchSize):
    # col is in the x-direction
    # row is in the y-direction
    # origin (0,0) is the top left corner of the image
    if row == 1:
        y = patchSize/2
    if col == 1:
        x = patchSize/2
    if row > 1:
        y = ((row-1)*patchSize) + (0.5*patchSize)
    if col > 1:
        x = ((col-1)*patchSize) + (0.5*patchSize)
    return x, y

def get_tiles(slide, patchSize, sample_images):
    
    tiles = DeepZoomGenerator(slide, tile_size=patchSize, overlap=0, limit_bounds=False)
    max_level = len(tiles.level_dimensions)-1   # highest resolution level

    cols, rows = tiles.level_tiles[max_level]

    tiled_images = []
    cols_rows_loc = []
    limit_no_images = sample_images

    for row in range(rows):
        for col in range(cols):
            temp_tile = tiles.get_tile(max_level, (col, row)) 
            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            if (temp_tile.size == (128,128)) and (130 < temp_tile_np.mean() < 170) and (temp_tile_np.std() > 15):
                tiled_images.append(temp_tile_np)
                x, y = get_center_of_tiles(col, row, patchSize)
                cols_rows_loc.append([x, y, (col,row)]) 
            if limit_no_images is None:
                continue
            elif len(tiled_images) >= limit_no_images:
                break
    return tiled_images, cols_rows_loc

# Below function returns a numpy array containing tiles with mitotic figures detected 
def detect_mitosis(slide, patchSize, sample_images, Conf, IoU):
    model_path = './yolov5/runs/train/MEL_128x128_scratch/weights/best.pt'
    model = torch.hub.load('./yolov5','custom', path=model_path, source='local')
    model.conf = Conf  # NMS confidence threshold
    model.iou = IoU  # NMS IoU threshold
    
    tiles = DeepZoomGenerator(slide, tile_size=patchSize, overlap=0, limit_bounds=False)
    max_level = len(tiles.level_dimensions)-1   # highest resolution level

    cols, rows = tiles.level_tiles[max_level]
    
    mitotic_images = []
    mitotic_pandas = pd.DataFrame()
    mitotic_cols_rows_loc = []
    limit_stop = sample_images

    for row in range(rows):
        for col in range(cols):
            temp_tile = tiles.get_tile(max_level, (col, row)) 
            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            if (temp_tile.size == (128,128)) and (130 < temp_tile_np.mean() < 170) and (temp_tile_np.std() > 15):
                detection_result = model(temp_tile_np, size=128)
                #print(detection_result)
                if (detection_result.pandas().xyxy[0]['name'] == 'Mitosis').values.any():
                    label = 'col_row_'+str(col)+'_'+str(row)  # this label is used for identification of tile by column (x-direction) and row (y-direction)
                    mitotic_images.append(temp_tile_np)       # numpy array of images saved for later view
                    temp1 = detection_result.pandas().xyxy[0]
                    temp1 = temp1[temp1['name']=='Mitosis'] # here filtering only the mitotic figures and dropping nonmitotic detection 
                    no_of_detection = len(temp1)  # total number of detected mitotic figure in a tile (reason for this is that based on confidence and IoU, the # might increase)
                    temp1['cols_location'], temp1['rows_location'], temp1['label'], temp1['detected_total'] = col, row, label, no_of_detection
                    mitotic_pandas = pd.concat([mitotic_pandas, temp1], axis=0)  # holds information about bounding box, tile location
                    x, y = get_center_of_tiles(col,row, patchSize)
                    mitotic_cols_rows_loc.append([x, y, (col,row)])
                    #detection_result.save(save_dir='runs/detect/exp/', exist_ok=True)


            if limit_stop is None:
                continue
            elif len(mitotic_images) >= limit_stop:
                break
    
    return mitotic_images, mitotic_pandas

# Below function builds a pandas dataframe with mitotic figure coordinates that then used to draw boxes on the whole slide image
def get_mitosis_box_location(mitotic_pandas_pd, patchSize):
    global_mitotic_map = pd.DataFrame()
    global_mitotic_map['label'] = mitotic_pandas_pd['label']
    global_mitotic_map['confidence'] = mitotic_pandas_pd['confidence']
    global_mitotic_map['tile_origin_x'] = (mitotic_pandas_pd['cols_location']-0.5)*patchSize  # individual tile top left origin in x direction
    global_mitotic_map['tile_origin_y'] = (mitotic_pandas_pd['rows_location']-0.5)*patchSize  # individual row top left origin in y direction
    global_mitotic_map['mitosis_coord_x_min'] = global_mitotic_map['tile_origin_x'] + mitotic_pandas_pd['xmin']  # bounding box top left coordinate in x direction
    global_mitotic_map['mitosis_coord_y_min'] = global_mitotic_map['tile_origin_y'] + mitotic_pandas_pd['ymin']  # bounding box top left coordinate in x direction
    global_mitotic_map['mitosis_coord_x_max'] = global_mitotic_map['tile_origin_x'] + mitotic_pandas_pd['xmax']  # bounding box top left coordinate in x direction
    global_mitotic_map['mitosis_coord_y_max'] = global_mitotic_map['tile_origin_y'] + mitotic_pandas_pd['ymax']  # bounding box top left coordinate in x direction
    return global_mitotic_map


# Below function is used to draw bounding boxes around mitotic figures on the whole slide image
def draw_global_box(slide, coordinates, patchSize=128, scf=0.02):
    """
    Draw boxes on the whole slide image
    :param slide : whole slide image
    :param coordinates : mitotic figure coordinates
    :patchSize : tile patch size
    :scf : to scale down whole slide image for viewing
    :return : PIL Image object with rectangles
    """
    slide_y = slide.dimensions[0]  # vertical size
    slide_x = slide.dimensions[1]  # horizontal size

    slide_thumb = slide.get_thumbnail(size=(scf*slide_y,scf*slide_x)) # this is a PIL image

    global_coordinates = get_mitosis_box_location(coordinates,patchSize)


    box = ImageDraw.Draw(slide_thumb)

    for i in range(len(global_coordinates)):
        x_min = scf*global_coordinates.iloc[i]['mitosis_coord_x_min']
        y_min = scf*global_coordinates.iloc[i]['mitosis_coord_y_min']
        x_max = scf*global_coordinates.iloc[i]['mitosis_coord_x_max']
        y_max = scf*global_coordinates.iloc[i]['mitosis_coord_y_max']
        conf = global_coordinates.iloc[i]['confidence']
        box.rectangle([x_min, y_min, x_max, y_max], outline='green', width=2)
        #box.text((x_min, y_min-10), str(round(conf)), fill='blue')  # too small to show up in the slide
    
    return slide_thumb

# Draw bounding boxes and confidence scores on tiled images
def draw_box(img, boxes):
    """
    Draw boxes on the picture
    :param img : PIL Image object
    :param boxes : numpy array of size [number_of_boxes, 6]
    :return : PIL Image object with rectangles
    """

    img = Image.fromarray(img)

    box = ImageDraw.Draw(img)
    box.rectangle([boxes[0], boxes[1], boxes[2], boxes[3]], outline ="blue", width=2)

    text = str(round(boxes[4],2))
    box.text((boxes[0],boxes[1]-10), text, fill='blue')
    
    return img.resize((256,256))
    