import tensorflow as tf
import os

from fnmatch import filter
from os.path import isdir, join
import matplotlib.pyplot as plt
import numpy as np

def est_f1(prec, recall):
    return (2*prec*recall) / (prec+recall)

def visualize_imgs(save_path, img_list, img_name):
    x_batch = img_list[0]
    batch_size = x_batch.shape[0]

    vis_img = None
    denominator = 1
    edge = x_batch[0].shape[0]
    step = edge + 3

    if edge == 32:
        vis_img = np.zeros(((step) * batch_size, (step) * len(img_list), 3))
        denominator = 1
    elif edge == 28:
        vis_img = np.zeros(((step) * batch_size, (step) * len(img_list)))
    # Warning: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). May need to be fixed.
    elif edge == 64:
        print(x_batch[0])
        vis_img = np.zeros(((step) * batch_size, (step) * len(img_list), 3)).astype(np.float32)
        denominator = 1  #now the imagenet range from 0 to 1
    else:
        raise

    for jj, x in enumerate(img_list):
        for ii in range(batch_size):
            vis_img[(ii) * step:(ii) * step + edge, step*jj:step*jj+edge] = x[ii] * 1.0 / denominator

    # vis_img[(ii) * step:(ii) * step + edge, step*4:step*4 + edge] = (x_batch_2[ii] - x_batch_neg[ii]) / denominator

    # print('max', np.max(np.abs(x_batch_2[ii] - x_batch_neg[ii])))
    dpi = 80
    figsize = vis_img.shape[1] * 1.0 / dpi, vis_img.shape[0] * 1.0 / dpi
    fig = plt.figure(figsize=figsize)

    plt.imshow(vis_img)
    # plt.show()
    fig.savefig(os.path.join(save_path, 'plot_{}.png'.format(img_name)), dpi=dpi)

def trainable_in(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def remove_duplicate_node_from_list(A, B):
    result = A
    for EB in B:
        flag=True
        for EA in A:
            if EB == EA:
                # print('find duplicate', EA)
                flag=False
                break
        if flag:
            result.append(EB)
    return result

def l2_norm_reshape(a):
    a, len = reshape_cal_len(a)
    a_norm = tf.reduce_sum(a * a + 1e-20, axis=1) ** 0.5
    b = a / tf.expand_dims(a_norm, 1)
    return b

def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns





def reshape_cal_len(x):
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
        prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    return x, prod_non_batch_dimensions







def get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds):
    m11 = len(top_conf_inds & top_bias_inds)
    m10 = len(top_conf_inds & bottom_bias_inds)
    m01 = len(bottom_conf_inds & top_bias_inds)
    m00 = len(bottom_conf_inds & bottom_bias_inds)



    precision = 0
    recall = 0
    fpr = 0
    F1 = 0
    if m01 + m11 > 0:
        precision = m11 / (m01 + m11)
    if m10 + m11 > 0:
        recall = m11 / (m10 + m11)
    if m00 + m01 > 0:
        fpr = m01 / (m00 + m01)
    if  precision+recall > 0:
        F1 = 2*precision*recall/(precision+recall)

    return precision, recall, fpr, m11, m01


def get_single_precision_recall_fpr(y, y_pred, cutoff_y, cutoff_y_pred=None):
    '''
    predicted_label_conf_bias_list:  A ndarray with size 2 * number consisting of the confusion and bias value for every pair.
    conf_top_percentage: the ground-truth confusion percentage threshold.
    top_percentages: x-axis percentage currently used.
    '''
    if not cutoff_y_pred:
        cutoff_y_pred = cutoff_y

    n = y.shape[0]

    top_conf_inds = set(np.argwhere(y<cutoff_y).flatten())
    bottom_conf_inds = set(np.argwhere(y>=cutoff_y).flatten())
    top_bias_inds = set(np.argwhere(y_pred<cutoff_y_pred).flatten())
    bottom_bias_inds = set(np.argwhere(y_pred>=cutoff_y_pred).flatten())

    top_bias_inds_r = set(np.random.choice(n, len(top_bias_inds)))
    bottom_bias_inds_r = set([i for i in range(n)]) - set(top_bias_inds)



    precision, recall, fpr, m11, m01 = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds)
    precision_r, recall_r, fpr_r, m11_r, m01_r = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds_r, bottom_bias_inds_r)
    return precision, recall, fpr, m11, m01, precision_r, recall_r, fpr_r, m11_r, m01_r
