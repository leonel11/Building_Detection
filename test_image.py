import os
import scipy.misc
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import aux_metrics
import cnn_models
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.models import model_from_json


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

SAMPLE_SIDE = 512
MODEL_ARCHITECTURE = '_UNet_architecture.json'
MODEL_WEIGHTS = '_UNet_weights.h5'
THRESHOLD = 0.5

OPTIMIZER = Adam()
model = model_from_json(open(MODEL_ARCHITECTURE).read())
model.load_weights(MODEL_WEIGHTS)
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy',
              metrics=['accuracy', aux_metrics.jaccard_idx, aux_metrics.sorensen_dice_coef])


def init_argparse():
    parser = ArgumentParser(description='Test satellite images on slices')
    parser.add_argument(
        '-f',
        '--file',
        nargs='?',
        help='file name',
        type=str)
    return parser


def make_prediction(img_file):
    img = Image.open(img_file)
    h_count, w_count = img.size[0] // SAMPLE_SIDE, img.size[1] // SAMPLE_SIDE
    res = np.zeros((h_count*SAMPLE_SIDE, w_count*SAMPLE_SIDE, 1))
    for h in range(h_count):
        for w in range(w_count):
            i, j = h*SAMPLE_SIDE, w*SAMPLE_SIDE
            img_slice = img.crop((i, j, i+SAMPLE_SIDE, j+SAMPLE_SIDE))
            slice_arr = np.asarray(img_slice, dtype='float32') / 255.0
            slice_arr = slice_arr[np.newaxis, ...]
            predictions = model.predict(slice_arr)
            res[j:j+SAMPLE_SIDE,i:i+SAMPLE_SIDE] += predictions[0]
    for h in range(h_count-1):
        for w in range(w_count-1):
            i, j = h*SAMPLE_SIDE+SAMPLE_SIDE//2, w*SAMPLE_SIDE+SAMPLE_SIDE//2
            img_slice = img.crop((i, j, i+SAMPLE_SIDE, j+SAMPLE_SIDE))
            slice_arr = np.asarray(img_slice, dtype='float32') / 255.0
            slice_arr = slice_arr[np.newaxis, ...]
            predictions = model.predict(slice_arr)
            res[j:j+SAMPLE_SIDE,i:i+SAMPLE_SIDE] += predictions[0]
            res[j:j+SAMPLE_SIDE,i:i+SAMPLE_SIDE] /= 2.0
    np.place(res, res>=THRESHOLD, [1.0])
    np.place(res, res<THRESHOLD, [0.0])
    im_res = np.array(255*res, dtype=np.int)
    scipy.misc.imsave('res_.jpg', im_res[:,:,0])


def main():
    args = init_argparse().parse_args()
    img_file = args.file
    make_prediction(img_file)


if __name__ == '__main__':
    main()
