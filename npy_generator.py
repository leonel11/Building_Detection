import numpy as np
import os
from PIL import Image


IMAGES_DIR =  'data_transformed/images'
MASKS_DIR = 'data_transformed/masks'
DATA_DIR = 'data_transformed'

SAMPLE_SIDE = 256


def get_all_images(path_to_dir):
    res = []
    for root, _, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith('.jpg'):
                res.append(os.path.join(root, file))
    return res


def create_samples(x_files, y_files, x_dir='x', y_dir='y', STAGE=0, add_axis=False):
    x_samples, y_samples = [], []
    w_count = 33 #WIDTH//SAMPLE_SIDE
    h_count = 10 #HEIGHT//SAMPLE_SIDE
    for idx in range(len(x_files)):
        x_img = Image.open(x_files[idx])
        y_img = Image.open(y_files[idx])
        x_data = np.asarray(x_img)
        y_data = np.asarray(y_img)
        for h in range(h_count):
            for w in range(w_count):
                i = h*SAMPLE_SIDE
                j = w*SAMPLE_SIDE
                x_matr = x_data[i:i+SAMPLE_SIDE, j:j+SAMPLE_SIDE]
                y_matr = y_data[i:i+SAMPLE_SIDE, j:j+SAMPLE_SIDE]
                if (np.amax(y_matr) > 0):
                    for iter in range(8):
                        x_im = Image.fromarray(x_matr)
                        y_im = Image.fromarray(y_matr)
                        x_im.save('data_transformed/{}/{:0>6}.jpg'.format(x_dir, STAGE))
                        y_im.save('data_transformed/{}/{:0>6}.jpg'.format(y_dir, STAGE))
                        x_samples.append(x_matr)
                        y_samples.append(y_matr)
                        STAGE = STAGE+1
                        if iter == 3:
                            x_matr = np.flip(x_data[i:i+SAMPLE_SIDE, j:j+SAMPLE_SIDE], 1)
                            y_matr = np.flip(y_data[i:i + SAMPLE_SIDE, j:j + SAMPLE_SIDE], 1)
                        else:
                            x_matr = np.rot90(x_matr)
                            y_matr = np.rot90(y_matr)
    if add_axis:
        return np.asarray(x_samples), np.asarray(y_samples)[..., np.newaxis]
    return np.asarray(x_samples), np.asarray(y_samples)


STAGE = 0
img_files = get_all_images(IMAGES_DIR)
mask_files = get_all_images(MASKS_DIR)
print(len(mask_files))
#X_train, Y_train = create_samples(img_files[1:], mask_files[1:], 'x_train', 'y_train', STAGE, True)
#X_test, Y_test = create_samples([img_files[0]], [mask_files[0]], 'x_test', 'y_test', STAGE, True)
#np.save(os.path.join(DATA_DIR, 'x_train.npy'), X_train)
#np.save(os.path.join(DATA_DIR, 'y_train.npy'), Y_train)
#np.save(os.path.join(DATA_DIR, 'x_test.npy'), X_test)
#np.save(os.path.join(DATA_DIR, 'y_test.npy'), Y_test)
cr = 1.0
