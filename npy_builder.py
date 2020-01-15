# Скрипт для формирования выборки 
import numpy as np
from PIL import Image
import os
import utils
import shutil


IMG_PATH = 'data/Sat/img'
MASK_PATH = 'data/Sat/masks'
NPY_PATH = 'data/Sat'
IMG_SAMPLES = 'data/Sat/samples_img'
MASK_SAMPLES = 'data/Sat/samples_masks'

SAMPLE_SIDE = 5656


def clean_slicing_folders():
    '''
    Очистка папок с нарезанными картинками и масками
    '''
    if os.path.exists(IMG_SAMPLES):
        shutil.rmtree(IMG_SAMPLES)
    os.makedirs(IMG_SAMPLES)
    if os.path.exists(MASK_SAMPLES):
        shutil.rmtree(MASK_SAMPLES)
    os.makedirs(MASK_SAMPLES)


def slice_samples(img_files, img_masks):
    '''
    Нарезка спутниковых снимков и масок на фрагменты меньшего размера (такие картинки требуются на вход сверточной нейронной сети)
    @param спутниковые снимки
    @param бинарные маски
    @return набор нарезанных спутниковых снимков и масок в виде 2 numpy-массивов
    '''
    x_samples, y_samples = [], []
    STAGE = 0
    print('Slice data:')
    for idx in range(len(filenames)):  # slicing
        img = Image.open(img_files[idx])
        mask = Image.open(mask_files[idx])
        h_count, w_count = img.size[0]//SAMPLE_SIDE, img.size[1]//SAMPLE_SIDE # count of slices on each dim
        for h in range(h_count):
            for w in range(w_count): # slice part of image and mask
                i, j = h*SAMPLE_SIDE, w*SAMPLE_SIDE
                img_slice = img.crop((i, j, i+SAMPLE_SIDE, j+SAMPLE_SIDE))
                mask_slice = mask.crop((i, j, i+SAMPLE_SIDE, j+SAMPLE_SIDE))
                img_slice.save(os.path.join(IMG_SAMPLES, filenames[idx] + '_{:0>5}.jpg'.format(STAGE)))
                mask_slice.save(os.path.join(MASK_SAMPLES, filenames[idx] + '_{:0>5}.png'.format(STAGE)))
                STAGE = STAGE + 1
                x_samples.append(np.asarray(img_slice))
                y_samples.append(np.asarray(mask_slice))
        print('{}/{} samples are sliced'.format(idx+1, len(filenames)))
    print('Slicing are finished!')
    return np.asarray(x_samples), np.asarray(y_samples)[..., np.newaxis]


def save_npy_samples(x_samples, y_samples):
    '''
    Сохранение выборок в npy-файлах
    @param x_samples 
    @param y_samples
    '''
    np.save(os.path.join(NPY_PATH, 'x_samples.npy'), x_samples)
    np.save(os.path.join(NPY_PATH, 'y_samples.npy'), y_samples)
    print('NPY-files are saved!')


filenames = utils.get_all_filenames(MASK_PATH)
print('Images for slicing:\t{}'.format(len(filenames)))
img_files = list(map(lambda s: os.path.join(IMG_PATH, s+'.jpg'), filenames)) # список спутниковых снимков
mask_files = list(map(lambda s: os.path.join(MASK_PATH, s+'.png'), filenames)) # список соответствующих бинарных масок
clean_slicing_folders()
x_samples, y_samples = slice_samples(img_files, mask_files)
print('NPY-files are formed!')
save_npy_samples(x_samples, y_samples)