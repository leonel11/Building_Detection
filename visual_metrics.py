import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

VISUAL_DATA = '../visual'

mpl.rcParams['font.size'] = 15
mpl.rc('lines', linewidth=3)


def _smooth_metric(metr):
    for idx, val in enumerate(metr):
        if val < 0.0:
            if len(metr) > 1:
                if idx == 0:
                    metr[idx] = metr[idx+1]
                    continue
                if idx == len(metr)-1:
                    metr[idx] = metr[idx-1]
                    continue
                metr[idx] = (metr[idx-1] + metr[idx+1])/ 2.0
            else:
                metr[idx] = 0.0
    return metr


def _visualize_metric(net_name, metric_name, epochs, train_metric, val_metric):
    plt.clf()
    plt.figure(figsize=(18,12))
    plt.grid(True)
    plt.title(net_name)
    plt.xlabel('epochs')
    plt.ylabel(metric_name)
    plt.plot(epochs, train_metric, label='train')
    plt.plot(epochs, val_metric, label='val')
    plt.legend()
    plt.savefig(os.path.join(VISUAL_DATA, '{}__{}.png'.format(net_name, metric_name)))


def plot_history(history, net_name, epochs_count):
    if not os.path.exists(VISUAL_DATA):
        os.makedirs(VISUAL_DATA)
    _visualize_metric(net_name, 'Loss', list(range(1, epochs_count+1)),
                      _smooth_metric(history.history['loss']),
                      _smooth_metric(history.history['val_loss']))
    _visualize_metric(net_name, 'Accuracy', list(range(1, epochs_count + 1)),
                      _smooth_metric(history.history['acc']),
                      _smooth_metric(history.history['val_acc']))
    _visualize_metric(net_name, 'Jaccard_Index', list(range(1, epochs_count + 1)),
                      _smooth_metric(history.history['jaccard_idx']),
                      _smooth_metric(history.history['val_jaccard_idx']))
    _visualize_metric(net_name, 'Sorensen_Dice_Coef', list(range(1, epochs_count + 1)),
                      _smooth_metric(history.history['sorensen_dice_coef']),
                      _smooth_metric(history.history['val_sorensen_dice_coef']))