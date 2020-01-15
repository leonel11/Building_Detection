# Скрипт для подсчета метрик качества сегментации
from keras import backend as K


K.set_image_dim_ordering('tf') # установка порядка измерений для картинок как в бэкэнде Tensorflow

SMOOTH = 1e-12


def jaccard_idx(y_true, y_pred):
	'''
	Подсчет индекса Жаккара
	@param y_true истинные результаты
	@param y_pred результаты, полученные с помощью алгоритма глубокого МО
	@return индекс Жаккара
	'''
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return K.mean(jac)


def sorensen_dice_coef(y_true, y_pred):
	'''
	Подсчет коэффициента Серенсена-Дайса
	@param y_true истинные результаты
	@param y_pred результаты, полученные с помощью алгоритма глубокого МО
	@return коэффициент Серенсена-Дайса
	'''
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (2.0*intersection + SMOOTH) / (sum_ + SMOOTH)
    return K.mean(jac)