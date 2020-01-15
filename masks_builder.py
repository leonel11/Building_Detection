# Скрипт для построения масок к спутниковым снимкам
from PIL import Image, ImageDraw
import os
import json
import shutil
import utils
import numpy as np


# Папки с данными
IMG_PATH = 'raw_data/Sat/img'
JSON_PATH = 'raw_data/Sat/ann'
MASK_PATH = 'C:/_Repositories/Building_Detection/raw_data/Sat/masks'


def read_marking(json_file):
    '''
    Считывание разметки из json-файла
    @param json_file
    @return считаные данные из json-файла
    '''
    json_data = None
    with open(json_file) as f:
        json_data = json.load(f)
    return json_data


def get_marking_polygons(json_data):
    '''
    Считывание полигонов разметки из данных с json-файла
    @param json_data данные, считанные из json-файла
    @return набор полигонов
    '''
    polygons = []
    for mark_dict in json_data['objects']:
        pol = mark_dict['points']['exterior']
        polygons.append(sum(pol, []))
    return polygons


def get_mask(mask_size, polygons):
    '''
    Построение бинарной маски для спутникового снимка
    @param mask_size размер маски
    @param @polygons набор полигонов
    @return сгенерированная бинарнаяя маска в виде картинки
    '''
    mask = Image.new('1', mask_size)
    for pol in polygons:
        d = ImageDraw.Draw(mask)
        d.polygon(pol, fill=1)
    return mask


filenames = utils.get_all_filenames(JSON_PATH)
print('Images for marking:\t{}'.format(len(filenames)))
json_files = list(map(lambda s: os.path.join(JSON_PATH, s+'.json'), filenames)) # список всех json-файлов с разметкой
img_files = list(map(lambda s: os.path.join(IMG_PATH, s+'.jpg'), filenames)) # список всех спутниковых снимков для разметки
if os.path.exists(MASK_PATH):
    shutil.rmtree(MASK_PATH)
    print('Clean folder old masks')
os.makedirs(MASK_PATH)
print('Create masks:')
for idx in range(len(filenames)): # генерация масок
    img = Image.open(img_files[idx])
    polygons = get_marking_polygons(read_marking(json_files[idx]))
    mask = get_mask(img.size, polygons)
    mask.save(os.path.join(MASK_PATH, filenames[idx]+'.png'))
    print('{}/{} masks are formed'.format(idx + 1, len(filenames)))
print('Masks are generated and saved!')