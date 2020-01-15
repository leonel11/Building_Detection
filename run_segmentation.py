import numpy as np
import os
from glob import glob
from PIL import Image
import aux_metrics
import cnn_models
from keras.optimizers import Adam, SGD, RMSprop, Adagrad


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# параметры обучения сети
BATCH_SIZE = 36 # количество обучающих примеров, которые видит оптимизирующая функция перед обновлением ее весов
EPOCHES = 100 # число эпох (сколько раз обучающая выборка предъявляется модели)
VERBOSE = 1 # режим отображения процесса обучения
OPTIMIZER = Adam() # оптимизирующая функция


def get_batch(x_files, y_files, batch_size=None):
    while True:
        x, y = [], []
        if batch_size == None:
            indeces = list(range(len(x_files)))
        else:
            indeces = np.random.randint(len(x_files), size=batch_size)
        for i, idx in enumerate(indeces):
            x_img = np.asarray(Image.open(x_files[idx])).astype('float32')
            x_img /= 255.0
            y_img = np.asarray(Image.open(y_files[idx])).astype('int')
            y_img //= 255
            x.append(x_img)
            y.append(y_img)
        x_batch = np.asarray(x)
        y_batch = np.asarray(y)[..., np.newaxis]
        yield x_batch, y_batch


print('Loading data')
x_samples = np.load('sat_data/x_samples.npy')
y_samples = np.load('sat_data/y_samples.npy')

print('Data normalization')
x_samples = x_samples.astype('float32')
x_samples /= 255.0

model = cnn_models.DLinkNet()
model.summary()
print('Model:   {}\n'.format(model.name))

parallel_model = model
parallel_model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', 
                       metrics=['accuracy', aux_metrics.jaccard_idx, aux_metrics.sorensen_dice_coef])
history = parallel_model.fit(x_samples, y_samples, batch_size=BATCH_SIZE, epochs=EPOCHES, verbose=VERBOSE,
                             shuffle=True, validation_split = 0.2) # обучение модели
score = model.evaluate(x_samples, y_samples, verbose=VERBOSE) # тестирование модели
print()
print('\nTest score:', score[0]) # значение функции потерь
print('Test accuracy:    ', score[1])
print('Jaccard index:    ', score[2])
print('Sorensen-Dice coefficient:    ', score[3])

# сохранение модели
model_json = model.to_json()
open(str(model.name) + '_architecture.json', 'w').write(model_json) # сохранение архитектуры
model.save_weights(str(model.name) + '_weights.h5', overwrite=True) # сохранение весов
