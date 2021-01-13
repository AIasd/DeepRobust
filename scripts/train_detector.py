# 1.Set test_save_neighbor_pred to True for saving neighbor's embedding for analysis,
# 2.Change train_bin_quantile and test_bin_quantile to 0.1 if 10% setting is wanted.
# 3.Set retrain_with_weak_points_for_detector to True and exp_num
# (coressponding to m in the paper) to reweight each data point's weight
# when sampling from them for training the detector.

import json
import argparse
import os
import pickle
import itertools
from time import time


import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn import metrics
from sklearn import linear_model

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import dataloader.loader

# from learning.model_mnist_large import ModelMNIST

from learning.detector import ModelDetector

from utils_new import trainable_in, remove_duplicate_node_from_list, l2_norm_reshape, visualize_imgs, get_single_precision_recall_fpr_helper, est_f1




def bin(neighbor_avg_acc, quantile=0.1, cutoff=None):
    # n_avg_acc = np.floor(neighbor_avg_acc * bin_number).astype(np.int64)
    # n_avg_acc = np.squeeze(n_avg_acc)
    if not cutoff:
        cutoff = np.quantile(neighbor_avg_acc, quantile)
    neighbor_avg_acc[neighbor_avg_acc > cutoff] = 1
    neighbor_avg_acc[neighbor_avg_acc <= cutoff] = 0
    neighbor_avg_acc = 1 - neighbor_avg_acc
    return neighbor_avg_acc


def get_embed_and_correctness(dataload, num_nat_batches=10, batch_size=50, model_load_dir='../load/cifar10', layer='x4', mode='plain', eval_data='test', prefix='0', vis_mode=False, angle=0, x_shift=0, y_shift=0, architecture='resnet', dataset_type='cifar10', save_input=False):
    '''
    Get the layer embeddings of the input data and the model's correctness on them.

    number of natural points = num_nat_batches * batch_size

    INPUT:
    num_nat_batches: number of batches to process.
    batch_size:
    model_load_dir: path to the folder containing the trained model.
    layer: ['x1', 'x2', 'x3', 'x4'] the layer to get.
    mode: ['plain', 'augmentation', 'rot_and_shift', 'rot_and_shift_exact']
    eval_data: ['train', 'test']
    prefix: used to differentiate visualization across different runs.
    vis_mode: if in a toy mode.
    angle: activated if mode in ['rot_and_shift', 'rot_and_shift_exact']
    x_shift: activated if mode in ['rot_and_shift', 'rot_and_shift_exact']
    y_shift: activated if mode in ['rot_and_shift', 'rot_and_shift_exact']
    architecture: name of the classificatino model used.
    transformed_center_load: If using transformed center images data loader.
    save_input: if to save the transformed images.

    OUTPUT:
    saved_transformed_images:
    natural_embed: 2d numpy array with size [number of natural points, embedding size] consists of the embedding at a certain layer when given input.
    y_vec: 1d numpy array with size [number of natural points,] consists of the ground-truth label of each point.
    y_pred_vec: 1d numpy array with size [number of natural points,] consists of the model's predicted label of each point.
    natural_correctness: 1d numpy array with size [number of natural points,] consists of the model's classification correctness of each data point when given input.
    '''

    model = None
    # Can customize the model used for cifar100 and svhn later
    if architecture == 'resnet':
        from learning.model_resnet import ModelVani
    elif architecture == 'wrn':
        from learning.model_wrn import ModelVani
    elif architecture == 'vgg16':
        from learning.model_vgg16 import ModelVani
    else:
        raise

    model = ModelVani(dataset_type, retrain_with_weak_points=False)

    print('restore model at :', model_load_dir)


    if dataset_type in ['cifar10', 'svhn', 'celeb5', 'celebA']:
        input_shape = [None, 32, 32, 3]
    elif dataset_type == 'fmnist':
        input_shape = [None, 28, 28, 1]
    else:
        raise

    # Construct variables and graph
    with tf.variable_scope('input'):
        # We do not train the classifier in this file so we can feed in a constant to this field.
        is_training = tf.constant([False], dtype=tf.bool)
        sample_prob_tf = tf.placeholder(tf.float32, shape=[batch_size, 1])
        images = tf.placeholder(tf.float32, shape=input_shape)
        labels = tf.placeholder(tf.int64, shape=None)
    print(mode, angle, x_shift, y_shift)
    transformed_images = dataloader.loader.transform(images, mode=mode, angle=angle, x_shift=x_shift, y_shift=y_shift)
    # Complete the graph
    layer_values_A, _, _, _, n_Aaccuracy, n_Apredict, mask = model._encoder(transformed_images, labels, is_training, sample_prob_tf)

    assert layer in ['x1', 'x2', 'x3', 'x4']
    f_feature = l2_norm_reshape(layer_values_A[layer])
    softmax = layer_values_A['softmax']

    # Create Saver
    var_main_encoder = trainable_in('main_encoder')
    var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
    restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
    saver_restore = tf.train.Saver(restore_var_list)

    saved_transformed_images = []
    natural_embed = []
    softmax_all = []
    y_vec = []
    y_pred_vec = []
    natural_correctness = []
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        # Restore trained model
        print('restore model at :', model_load_dir)

        model_dir_load = tf.train.latest_checkpoint(model_load_dir)
        saver_restore.restore(sess, model_dir_load)

        pred_acc = 0
        dt = []
        for ii in range(num_nat_batches):
            x_batch, y_batch = None, None
            if eval_data == 'train':
                x_batch, y_batch = dataload.train_data.get_next_batch(batch_size, multiple_passes=True, reshuffle_after_pass=False)
            else:
                x_batch, y_batch = dataload.eval_data.get_next_batch(batch_size, multiple_passes=True, reshuffle_after_pass=False)

            sample_prob_slice_constant = np.ones([batch_size, 1])
            import time
            t1 = time.time()
            images_vis = sess.run([transformed_images], feed_dict={sample_prob_tf: sample_prob_slice_constant, images: x_batch, labels: y_batch})
            t2 = time.time()
            dt.append(t2-t1)
            images_vis, f_feature_eval, softmax_eval, nat_acc, prediction_correctness, y, y_pred = sess.run([transformed_images, f_feature, softmax, n_Aaccuracy, mask, labels, n_Apredict], feed_dict={sample_prob_tf: sample_prob_slice_constant, images: x_batch, labels: y_batch})

            # Save information
            pred_acc += nat_acc
            natural_embed.append(f_feature_eval)
            softmax_all.append(softmax_eval)
            y_vec.append(y)
            y_pred_vec.append(y_pred)
            natural_correctness.append(prediction_correctness)

            if save_input:
                saved_transformed_images.append(images_vis)

            if ii % 100 == 0 and ii > 0:
                print('inner', ii, '/', num_nat_batches, pred_acc / ii)
            # print(predictions, labels_truth)

            # Visualization
            if vis_mode == True:
                tmp_img_folder = 'tmp_img'
                if not os.path.exists(tmp_img_folder):
                    os.mkdir(tmp_img_folder)
                # print(images_vis)
                # print(np.max(images_vis), np.min(images_vis))
                visualize_imgs(tmp_img_folder+'/', [images_vis], mode+'_'+eval_data+'_'+prefix+'_'+str(ii))
        print(np.mean(dt), np.std(dt))
        print('classification accuracy at', prefix, ':', pred_acc / num_nat_batches)
        natural_embed = np.concatenate(natural_embed, axis=0)
        softmax_all = np.concatenate(softmax_all, axis=0)
        y_vec = np.concatenate(y_vec, axis=0)
        y_pred_vec = np.concatenate(y_pred_vec, axis=0)
        natural_correctness = np.concatenate(natural_correctness, axis=0)
        if save_input:
            saved_transformed_images = np.concatenate(saved_transformed_images, axis=0)

    tf.reset_default_graph()

    return saved_transformed_images, natural_embed, softmax_all, y_vec, y_pred_vec, natural_correctness


def get_avg_robustness(config, num_nat_batches, batch_size, model_load_dir, layer, transformation_mode, eval_data, sample_size, save_neighbor_pred, vis_mode, architecture, transform_center=False, save_input=False):
    '''
    OUTPUT:
    natural_embed: feature vector of all images (either training or testing).
    y: ground-truth labels of all images.
    y_pred: predicted labels of all images.
    avg_correctness: average accuracy of each image's neighbors. Note the value for this field for a neighbor of an image is the same as that for the image.
    '''



    natural_embed = None
    total_correctness = None
    print('-'*10, eval_data, 'plain', '-'*10)
    start = time()

    # Get info for the center images
    data_path = config['data_path']
    dload_transformed = dataloader.loader.build_loader(data_path, dataset_type, transform_center=transform_center)

    saved_transformed_images_0, natural_embed, softmax_all, y, y_pred, total_correctness = get_embed_and_correctness(dload_transformed, num_nat_batches=num_nat_batches, batch_size=batch_size, model_load_dir=model_load_dir, layer=layer, mode='plain', eval_data=eval_data, vis_mode=vis_mode, architecture=architecture, dataset_type=dataset_type, save_input=save_input)
    print(eval_data, 'plain', time()-start)
    natural_acc = np.mean(total_correctness)


    natural_embed_list = [natural_embed]
    y_list = [y]
    y_pred_list = [y_pred]
    saved_transformed_images_list = []

    if save_input:
        saved_transformed_images_list = [saved_transformed_images_0]

    # Angles to sweep
    # Note: angle and shift are only used when mode=='rot_and_shift_exact'

    assert transformation_mode in ['sweeping_rotate', 'sweeping_shift', 'sweeping_rotate_and_shift', 'sweeping_rotate_and_random_shift', 'sweeping_rotate_and_25_random_shift', 'random_rotate', 'random_shift', 'random_rotate_and_shift']

    mode = None
    angles = None
    x_shifts = None
    y_shifts = None

    # sample_size is fixed when sweeping is used
    if transformation_mode == 'sweeping_rotate':
        sample_size = 20
        angle_max = 30
        sample_size_half = sample_size // 2
        angle_step = angle_max / sample_size_half
        angles = [(-i-1)*angle_step for i in range(sample_size_half)] + [(i+1)*angle_step for i in range(sample_size_half)]

        x_shifts = [0 for _ in range(sample_size)]
        y_shifts = [0 for _ in range(sample_size)]
        mode = 'rot_and_shift_exact'
    elif transformation_mode == 'sweeping_shift':
        sample_size = 7 * 7
        a = [-3, -2, -1, 0, 1, 2, 3]
        b = [-3, -2, -1, 0, 1, 2, 3]
        x_shifts, y_shifts = list(zip(*list(itertools.product(a, b))))
        angles = [0 for _ in range(sample_size)]
        mode = 'rot_and_shift_exact'
    elif transformation_mode == 'sweeping_rotate_and_shift':
        sample_size = 5 * 5 * 25
        angles = [-30] * 25 + [-27.5] * 25 + [-25] * 25 + [-22.5] * 25 + [-20] * 25 + [-17.5] * 25 + [-15] * 25 + [-12.5] * 25 + [-10] * 25 + [-7.5] * 25 + [-5] * 25 + [-2.5] * 25 + [0] * 25 + [2.5] * 25 + [5] * 25 + [7.5] * 25 + [10] * 25 + [12.5] * 25 + [15] * 25 + [17.5] * 25 + [20] * 25 + [22.5] * 25 + [25] * 25 + [27.5] * 25 + [30] * 25
        a = [-3, -1.5, 0, 1.5, 3]
        b = [-3, -1.5, 0, 1.5, 3]
        x_shifts, y_shifts = list(zip(*(list(itertools.product(a, b)) * 25)))
        mode = 'rot_and_shift_exact'
    elif transformation_mode =='sweeping_rotate_and_random_shift':
        sample_size = 25 * 1
        angles = np.arange(-30, 32.5, 2.5).tolist() * 1
        x_shifts = np.random.rand(sample_size) * 1
        y_shifts = np.random.rand(sample_size) * 1
        mode = 'rot_and_shift_exact'
    elif transformation_mode =='sweeping_rotate_and_25_random_shift':
        sample_size = 25 * 25
        angles = np.arange(-30, 32.5, 2.5).tolist() * 25
        x_shifts = np.random.rand(sample_size)
        y_shifts = np.random.rand(sample_size)
        mode = 'rot_and_shift_exact'
    elif transformation_mode == 'random_rotate':
        angles = [30 for _ in range(sample_size)]
        x_shifts = [0 for _ in range(sample_size)]
        y_shifts = [0 for _ in range(sample_size)]
        mode = 'rot_and_shift'
    elif transformation_mode == 'random_shift':
        angles = [0 for _ in range(sample_size)]
        x_shifts = [3 for _ in range(sample_size)]
        y_shifts = [3 for _ in range(sample_size)]
        mode = 'rot_and_shift'
    elif transformation_mode == 'random_rotate_and_shift':
        angles = [30 for _ in range(sample_size)]
        x_shifts = [3 for _ in range(sample_size)]
        y_shifts = [3 for _ in range(sample_size)]
        mode = 'rot_and_shift'

    # print('angles :', angles)

    # Get info for the neighbor images
    for ii in range(sample_size):
        angle = angles[ii]
        x_shift = x_shifts[ii]
        y_shift = y_shifts[ii]

        # print('-'*10, eval_data, 'rot_and_shift', ii, '-'*10)
        saved_transformed_images_ii, natural_embed_ii, _, y_ii, y_pred_ii, correctness_ii = get_embed_and_correctness(dload_transformed, num_nat_batches=num_nat_batches, batch_size=batch_size, model_load_dir=model_load_dir, layer=layer, mode=mode, eval_data=eval_data, prefix=str(ii), vis_mode=vis_mode, angle=angle, x_shift=x_shift, y_shift=y_shift, architecture=architecture, dataset_type=dataset_type, save_input=save_input)

        total_correctness += correctness_ii
        print(ii, y_ii)


        if save_neighbor_pred:
            # Since we may want to estimate diversity score for training but
            # do not want the saved files to be too large.
            if eval_data == 'test':
                natural_embed_list.append(natural_embed_ii)
            if save_input:
                saved_transformed_images_list.append(saved_transformed_images_ii)
            y_list.append(y_ii)
            y_pred_list.append(y_pred_ii)

        print('time for', eval_data, ii, '/', sample_size, time()-start)

    avg_correctness = total_correctness / (sample_size + 1)
    print('robust accuracy :', np.mean(avg_correctness))
    print('natural accuracy :', natural_acc)
    if eval_data == 'train' and save_neighbor_pred:
        natural_embed = None
    else:
        natural_embed = np.concatenate(natural_embed_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    saved_transformed_images = None
    if save_input:
        saved_transformed_images = np.concatenate(saved_transformed_images_list, axis=0)

    return saved_transformed_images, natural_embed, softmax_all, y, y_pred, avg_correctness


def build_iter(embed, neighbor_avg_acc_input):
    '''
    Build tf.dataset about embed and neighbor_avg_acc_input to train/test the detector.
    '''
    images, labels = embed, neighbor_avg_acc_input
    x_shape = images.shape
    y_shape = labels.shape
    images_placeholder = tf.placeholder(images.dtype, x_shape)
    labels_placeholder = tf.placeholder(labels.dtype, y_shape)
    dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size
    )
    iter = dataset.make_initializable_iterator()
    return iter, images_placeholder, images, labels_placeholder, labels


def estimate_neighbor_acc_by_threshold(angle_threshold, y, y_pred):
    test_len = y.shape[0]
    neighbor_acc = np.zeros(test_len)
    neigh_num_p_1 = y_pred.shape[0] // test_len
    for i in range(neigh_num_p_1):
        start = i * test_len
        end = (i+1)*test_len
        neighbor_acc[(y_pred[start:end] - y)**2 < angle_threshold] += 1
    neighbor_acc /= neigh_num_p_1

    return neighbor_acc


def get_angle_threshold(path_train_embed_and_neighbor_acc):
    d = np.load(path_train_embed_and_neighbor_acc)
    train_y = d['train_y'][:37888]
    train_y_pred = d['train_y_pred'][:37888]
    lamb = 3
    angle_threshold = metrics.mean_squared_error(train_y, train_y_pred[:train_y.shape[0]]) * lamb
    print('angle_threshold :', angle_threshold)
    return angle_threshold




def est_f1(prec, recall):
    return (2*prec*recall) / (prec+recall)



















if __name__ == '__main__':

    # seeding randomness
    tf.compat.v1.set_random_seed(0)
    np.random.seed(0)

    # Parse input parameters
    parser = argparse.ArgumentParser(description='Train/Test Detector')
    parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--running_mode', dest='running_mode', type=str, default='full', help='running_mode to use')
    parser.add_argument('--architecture', dest='architecture', type=str, default='resnet', help='architecture to use')

    args = parser.parse_args()
    dataset_type = args.dataset
    architecture = args.architecture
    running_mode = args.running_mode

    assert dataset_type in ['cifar10', 'cifar100', 'svhn', 'fmnist', 'celeb5', 'celebA', 'simulation']
    assert architecture in ['resnet', 'wrn', 'vgg16', 'chauffeur', 'dave2', 'epoch']
    assert running_mode in ['full', 'preprocess', 'train_and_test', 'preprocess_train', 'preprocess_test', 'train', 'test', 'result', 'preprocess_train_and_train_and_test']
    if dataset_type in ['simulation']:
        assert running_mode in ['train_and_test', 'train', 'test']

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, visible_device_list='0')
    # gpu_options = tf.GPUOptions()

    with open('configs/config_'+dataset_type+'_detector.json') as config_file:
        config = json.load(config_file)


    # Type of model ['natural', 'madry', 'alp', 'triplet']. Currently, only 'natural' is supported
    model = 'natural'
    # Architecture ['resnet', 'wrn'， ‘vgg16’]

    batch_size = 100
    if dataset_type in ['cifar10', 'cifar100']:
        num_train_nat_batches = int(50000 / batch_size)
        num_test_nat_batches = int(10000 / batch_size)
    elif dataset_type in ['fmnist']:
        num_train_nat_batches = int(60000 / batch_size)
        num_test_nat_batches = int(10000 / batch_size)
    elif dataset_type in ['svhn']:
        num_train_nat_batches = int(np.ceil(73257 / batch_size))
        num_test_nat_batches = int(np.ceil(26032 / batch_size))
    elif dataset_type in ['celeb5']:
        batch_size = 5
        num_train_nat_batches = int(np.ceil(91 / batch_size))
        num_test_nat_batches = int(np.ceil(24 / batch_size))
    elif dataset_type in ['celebA']:
        num_train_nat_batches = int(np.ceil(2499 / batch_size))
        num_test_nat_batches = int(np.ceil(500 / batch_size))
    elif dataset_type in ['simulation']:
        num_train_nat_batches = int(np.ceil(37888 / batch_size))
        num_test_nat_batches = int(np.ceil(9472 / batch_size))

    # Number of neighbors used for each image
    # TBD: experiment 6, 12, 25, 50, 200 for 3 models+resnet
    num_of_transformation_for_train = 50
    num_of_transformation_for_test = 50
    train_detector_epoch = 200
    layer = 'x4'
    momentum = config['momentum']
    step_size_schedule = config['step_size_schedule']
    weight_decay = config['weight_decay']
    optimizer = config['optimizer']
    # optimizer = 'adam'
    n_jobs = 20

    # control whether run the full dataset or only several examples for debugging/visualizing generated images
    vis_mode = False
    # ['regression', 'classification']
    task_type = 'classification'
    # ['nn', 'svc', 'knn', 'lr']
    # nn, svc, knn for classification, nn, lr, knn for regression
    detector = 'nn'
    # creteria for detector: control center based (only use center image for training and testing classifier/regressor) or region based (use center images and all neighbor images for training and testing classifier/regressor)
    use_mask = False
    is_finetune = False

    # if save neighbors' predictions
    train_save_neighbor_pred = False
    test_save_neighbor_pred = False

    # TBD: fix "transform_center_train will also transform only test data so it is buggy.
    # it is not used however."
    transform_center_train = False
    transform_center_test = False


    # trade-off precision and recall
    # TBD: experiment 0, 1, 2 for 3 models+resnet
    retrain_with_weak_points_for_detector = False
    exp_num = 0
    if retrain_with_weak_points_for_detector:
        train_save_neighbor_pred = False
        test_save_neighbor_pred = False

    # positive integer, used if doing bin classification
    bin_number = 2

    train_cutoff = 0.5
    test_cutoff = train_cutoff
    # ['sweeping_rotate', 'sweeping_shift', 'sweeping_rotate_and_shift',
    # 'sweeping_rotate_and_random_shift',
    # 'sweeping_rotate_and_25_random_shift',
    # 'random_rotate', 'random_shift', 'random_rotate_and_shift']
    train_transformation_mode = 'random_rotate_and_shift'
    test_transformation_mode = train_transformation_mode
    if dataset_type in ['simulation']:
        train_transformation_mode = 'rainy_foggy_automold'
        test_transformation_mode = train_transformation_mode

    save_input_test = False


    save_input_test_str = ''
    if save_input_test:
        save_input_test_str = '_save_input'



    # Paths for saving data
    transformation_mode_str = '_'+train_transformation_mode
    if train_transformation_mode in  ['random_rotate_and_shift', 'rainy_foggy_automold']:
        transformation_mode_str += ('_'+str(num_of_transformation_for_train))
    test_transformation_mode_str = '_'+test_transformation_mode
    # The second condition is for backward compatability
    if test_transformation_mode in  ['random_rotate_and_shift', 'rainy_foggy_automold'] and num_of_transformation_for_test != 50:
        test_transformation_mode_str += ('_'+str(num_of_transformation_for_test))
    retrain_str = ''
    if use_mask:
        retrain_str = '_mask'

    retrain_with_weak_points_for_detector_str = ''
    if retrain_with_weak_points_for_detector:
        retrain_with_weak_points_for_detector_str = '_detector_mask'+'_'+str(exp_num)

    finetune_str = ''
    if is_finetune:
        finetune_str = '_finetune'

    tmp_data_folder = 'tmp_data_without_neighbor/'+dataset_type

    if test_save_neighbor_pred or train_save_neighbor_pred:
        tmp_data_folder = 'tmp_data_with_neighbor/'+dataset_type

    transform_center_test_str = ''
    if transform_center_test:
        transform_center_test_str = '_transform_center_test'


    train_cutoff_str = '_'+str(train_cutoff)


    # path to the classification model to be loaded
    model_load_dir = config['model_load_dir_ziyuan']+'/'+model+'_'+architecture+finetune_str+retrain_str

    # path to the detector model to be saved
    model_detector_dir = config['model_load_dir_ziyuan']+'/'+model+'_'+architecture+'_'+task_type+retrain_with_weak_points_for_detector_str+transformation_mode_str+finetune_str+retrain_str+train_cutoff_str+transform_center_test_str+'_detector'

    if not os.path.exists('tmp_data'):
       os.mkdir('tmp_data')
    if not os.path.exists(tmp_data_folder):
       os.mkdir(tmp_data_folder)
    if not os.path.exists(config['model_load_dir_ziyuan']):
       os.mkdir(config['model_load_dir_ziyuan'])
    if not os.path.exists(model_detector_dir):
       os.mkdir(model_detector_dir)

    vis_mode_str = ''
    if vis_mode:
        vis_mode_str = '_vis_mode'


    layer_str = '_'+layer


    non_nn_model_path = model_detector_dir+'/'+detector+'_'+architecture+vis_mode_str+'_'+layer_str+finetune_str+retrain_str+'.pkl'

    path_train_embed_and_neighbor_acc = tmp_data_folder+'/'+model+'_'+'train_embed_and_neighbor_acc'+'_'+architecture+transformation_mode_str+vis_mode_str+layer_str+finetune_str+retrain_str+'.npz'
    path_test_embed_and_neighbor_acc = tmp_data_folder+'/'+model+'_'+'test_embed_and_neighbor_acc'+'_'+architecture+test_transformation_mode_str+vis_mode_str+layer_str+finetune_str+retrain_str+transform_center_test_str+save_input_test_str+'.npz'

    path_detector_test_result = tmp_data_folder+'/'+model+'_'+task_type+'_'+'detector_test_result'+'_'+detector+'_'+architecture+transformation_mode_str+vis_mode_str+layer_str+retrain_with_weak_points_for_detector_str+finetune_str+retrain_str+'.npz'


    if dataset_type in ['simulation']:
        angle_threshold = get_angle_threshold(path_train_embed_and_neighbor_acc)

    if vis_mode == True:
        num_test_nat_batches = 1
        num_train_nat_batches = 2
        batch_size = 5
        num_of_transformation_for_train = 4
        num_of_transformation_for_test = 4
        # running_mode = 'preprocess'

    start = time()
    # get and save classifier's embedding for center images and average accuracy on their neighbors for train
    if running_mode in ['full', 'preprocess', 'preprocess_train', 'preprocess_train_and_train_and_test']:
        train_saved_transformed_images, train_embed, train_softmax, train_y_vec, train_y_pred_vec, train_neighbor_avg_acc = get_avg_robustness(config=config, num_nat_batches=num_train_nat_batches, batch_size=batch_size, model_load_dir=model_load_dir, layer=layer, transformation_mode=train_transformation_mode, eval_data='train', sample_size=num_of_transformation_for_train, save_neighbor_pred=train_save_neighbor_pred, vis_mode=vis_mode, architecture=architecture, transform_center=transform_center_train)

        train_saved_data = {'train_saved_transformed_images': train_saved_transformed_images, 'train_embed': train_embed, 'train_softmax': train_softmax, 'train_y': train_y_vec, 'train_y_pred': train_y_pred_vec, 'train_neighbor_avg_acc': train_neighbor_avg_acc}
        np.savez(path_train_embed_and_neighbor_acc, **train_saved_data)
        print('save train_embed_and_neighbor_acc at :', path_train_embed_and_neighbor_acc)
        # print('train_neighbor_avg_acc :', train_neighbor_avg_acc)

    # get and save classifier's embedding for center images and average accuracy on their neighbors for test
    if running_mode in ['full', 'preprocess', 'preprocess_test']:
        test_saved_transformed_images, test_embed, test_softmax, test_y_vec, test_y_pred_vec, test_neighbor_avg_acc = get_avg_robustness(config=config, num_nat_batches=num_test_nat_batches, batch_size=batch_size, model_load_dir=model_load_dir, layer=layer, transformation_mode=test_transformation_mode, eval_data='test', sample_size=num_of_transformation_for_test, save_neighbor_pred=test_save_neighbor_pred, vis_mode=vis_mode, architecture=architecture, transform_center=transform_center_test, save_input=save_input_test)


        test_saved_data = {'test_saved_transformed_images': test_saved_transformed_images, 'test_embed': test_embed, 'test_softmax':test_softmax, 'test_y': test_y_vec, 'test_y_pred': test_y_pred_vec, 'test_neighbor_avg_acc': test_neighbor_avg_acc}
        np.savez(path_test_embed_and_neighbor_acc, **test_saved_data)
        # print('test_neighbor_avg_acc : ', test_neighbor_avg_acc)

    # train detector
    if running_mode in ['full', 'train_and_test', 'train', 'preprocess_train_and_train_and_test']:
        # Preparing data for training
        train_loaded_data = np.load(path_train_embed_and_neighbor_acc)
        train_embed = train_loaded_data['train_embed']
        train_neighbor_avg_acc = train_loaded_data['train_neighbor_avg_acc']
        if dataset_type in ['simulation']:
            train_neighbor_avg_acc = np.abs(np.resize(train_loaded_data['train_y'], train_loaded_data['train_y'].shape[0]*(num_of_transformation_for_train+1)) - train_loaded_data['train_y_pred'])
        print('number of training points :',train_neighbor_avg_acc.shape[0])

        if dataset_type in ['simulation']:
            train_neighbor_avg_acc = estimate_neighbor_acc_by_threshold(angle_threshold, train_loaded_data['train_y'], train_loaded_data['train_y_pred'])
            print('train_neighbor_avg_acc:', train_neighbor_avg_acc[:100])

        # print(train_embed.shape, train_neighbor_avg_acc.shape)
        train_neighbor_avg_acc = np.tile(train_neighbor_avg_acc, (train_embed.shape[0]//train_neighbor_avg_acc.shape[0],))

        if task_type == 'classification':
            # test_loaded_data = np.load(path_test_embed_and_neighbor_acc)
            # test_neighbor_avg_acc = test_loaded_data['test_neighbor_avg_acc']
            # cutoff = np.quantile(test_neighbor_avg_acc, test_bin_quantile)

            train_neighbor_avg_acc = bin(train_neighbor_avg_acc, cutoff=train_cutoff).astype('int64')



        if detector != 'nn':
            clf = None
            if task_type == 'classification':
                if detector == 'svc':
                    clf = LinearSVC()
                elif detector == 'knn':
                    clf = KNeighborsClassifier(10, n_jobs=n_jobs)
            else:
                if detector == 'lr':
                    clf = linear_model.LinearRegression(n_jobs=n_jobs)
                elif detector == 'knn':
                    clf = KNeighborsRegressor(10, n_jobs=n_jobs)
            # print(train_embed.shape, train_neighbor_avg_acc.shape)
            print('train_embed.shape :', train_embed.shape)
            print('train_neighbor_avg_acc.shape :', train_neighbor_avg_acc.shape)
            clf.fit(train_embed, train_neighbor_avg_acc)

            with open(non_nn_model_path, 'wb') as fid:
                pickle.dump(clf, fid)

        else:
            num_of_training_points = train_neighbor_avg_acc.shape[0]
            num_of_steps_per_epoch = num_of_training_points // batch_size

            train_neighbor_avg_acc_input = train_neighbor_avg_acc
            if task_type == 'regression':
                train_neighbor_avg_acc_input = train_neighbor_avg_acc_input[:, np.newaxis]

            # Construct graph for training detector
            model = ModelDetector(task_type, bin_number)

            # Construct graph for training detector
            global_step = tf.train.get_or_create_global_step()

            images, labels = train_embed, train_neighbor_avg_acc_input
            x_shape = list(images.shape)
            x_shape[0] = None
            y_shape = list(labels.shape)
            y_shape[0] = None

            with tf.variable_scope('input'):
                is_training = tf.placeholder(tf.bool, shape=None)
                images_placeholder = tf.placeholder(images.dtype, x_shape)
                labels_placeholder = tf.placeholder(labels.dtype, y_shape)

            iter = dataloader.loader.DataSubset(train_embed, train_neighbor_avg_acc_input)


            _, mean_loss, weight_decay_loss, mean_error, _ = \
                model._encoder(images_placeholder, labels_placeholder, is_training)


            saver = tf.train.Saver(max_to_keep=3)
            var_main_encoder = trainable_in('detector_encoder')
            print("finish building up model")

            total_loss = mean_loss + weight_decay * weight_decay_loss

            encoder_opt = None
            boundaries = [int(sss[0]) for sss in step_size_schedule]
            boundaries = boundaries[1:]
            values = [sss[1] for sss in step_size_schedule]
            learning_rate = tf.train.piecewise_constant(
                tf.cast(global_step, tf.int32),
                boundaries,
                values)

            if optimizer == "SGD":
                encoder_opt = tf.train.MomentumOptimizer(learning_rate, momentum)
            elif optimizer == "adam":
                encoder_opt = tf.train.AdamOptimizer(learning_rate)

            grads1 = encoder_opt.compute_gradients(total_loss, var_list=var_main_encoder)

            train_step_m_encoder = encoder_opt.apply_gradients(grads1)

            new_global_step = tf.add(global_step, 1, name='global_step/add')
            increment_global_step_op = tf.assign(
                global_step,
                new_global_step,
                name='global_step/assign'
            )

            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                best_train_err = np.inf

                for i in range(train_detector_epoch):
                    cur_train_err = 0
                    for j in range(num_of_steps_per_epoch):
                        images_new, labels_new = None, None

                        if retrain_with_weak_points_for_detector:
                            d = np.load(path_train_embed_and_neighbor_acc)
                            images_new, labels_new = iter.get_next_batch_reweight(batch_size, d['train_neighbor_avg_acc'], exp_num=exp_num)
                        else:
                            images_new, labels_new = iter.get_next_batch(batch_size, reshuffle_after_pass=True)
                        _, _, cur_train_err_j = sess.run([increment_global_step_op, train_step_m_encoder, mean_error], feed_dict={is_training: True, images_placeholder: images_new, labels_placeholder: labels_new})
                        cur_train_err += cur_train_err_j
                    cur_train_err /= num_of_steps_per_epoch
                    print('train error at ', i, ':', cur_train_err)
                    if cur_train_err < best_train_err:
                        print('save model in ', model_detector_dir)
                        saver.save(sess, os.path.join(model_detector_dir, 'checkpoint'), global_step=global_step)
                        best_train_err = cur_train_err
            tf.reset_default_graph()

    # test detector
    if running_mode in ['full', 'train_and_test', 'test', 'preprocess_train_and_train_and_test']:
        # Preparing data for testing
        test_loaded_data = np.load(path_test_embed_and_neighbor_acc)
        test_embed = test_loaded_data['test_embed']
        test_neighbor_avg_acc = test_loaded_data['test_neighbor_avg_acc']
        if dataset_type in ['simulation']:
            test_neighbor_avg_acc = np.abs(np.resize(test_loaded_data['test_y'], test_loaded_data['test_y'].shape[0]*(num_of_transformation_for_test+1)) - test_loaded_data['test_y_pred'])

        print('number of testing points :',test_neighbor_avg_acc.shape[0])
        if dataset_type in ['simulation']:
            test_neighbor_avg_acc = estimate_neighbor_acc_by_threshold(angle_threshold, test_loaded_data['test_y'], test_loaded_data['test_y_pred'])
            print('test_neighbor_avg_acc:', test_neighbor_avg_acc[:100])


        print('-'*100, stats.percentileofscore(test_neighbor_avg_acc, 0.5), '-'*100)
        print('-'*100, stats.percentileofscore(test_neighbor_avg_acc, 0.75), '-'*100)

        test_neighbor_avg_acc = np.tile(test_neighbor_avg_acc, (test_embed.shape[0]//test_neighbor_avg_acc.shape[0],))




        if task_type == 'classification':
            test_neighbor_avg_acc = bin(test_neighbor_avg_acc, cutoff=test_cutoff).astype('int64')

        if detector != 'nn':
            with open(non_nn_model_path, 'rb') as fid:
                clf = pickle.load(fid)
            print(detector, 'classifier score :', clf.score(test_embed, test_neighbor_avg_acc))


            test_predictions_vec = clf.predict(test_embed)
            test_labels_vec = test_neighbor_avg_acc
            print('l1 error :', np.mean(np.abs(test_predictions_vec-test_labels_vec)))

        else:
            # print(test_neighbor_avg_acc.shape[0])
            test_total_steps = test_neighbor_avg_acc.shape[0] // batch_size
            if test_neighbor_avg_acc.shape[0] % batch_size > 0:
                test_total_steps += 1

            test_neighbor_avg_acc_input = np.copy(test_neighbor_avg_acc)
            if task_type == 'regression':
                test_neighbor_avg_acc_input = test_neighbor_avg_acc_input[:, np.newaxis]

            # Construct graph for testing detector
            model = ModelDetector(task_type, bin_number)

            global_step = tf.train.get_or_create_global_step()
            with tf.variable_scope('input'):
                is_training = tf.placeholder(tf.bool, shape=None)

            iter, images_placeholder, images, labels_placeholder, labels = build_iter(test_embed, test_neighbor_avg_acc_input)
            images_new, labels_new = iter.get_next()

            _, _, _, mean_error, predictions = \
                model._encoder(images_new, labels_new, is_training)

            # Restore trained model
            var_main_encoder = trainable_in('detector_encoder')
            var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='detector_encoder')
            restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
            saver_restore = tf.train.Saver(restore_var_list)



            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(iter.initializer, feed_dict={images_placeholder: images, labels_placeholder: labels})
                model_detector_dir = tf.train.latest_checkpoint(model_detector_dir)
                saver_restore.restore(sess, model_detector_dir)
                # test_labels_list = []
                test_err_list = []
                test_predictions_list = []
                for i in range(test_total_steps):

                    test_labels, cur_test_err, test_predictions = sess.run([labels_new, mean_error, predictions], feed_dict={is_training: False})
                    # test_labels_list.append(test_labels)
                    test_err_list.append(cur_test_err)
                    test_predictions_list.append(test_predictions)
                    # print('test error', i, ':', cur_test_err)

                # print('test_predictions :', test_predictions[:300])
                # print('test_labels :', test_labels[:300])

                print('average test error :', np.mean(test_err_list))
                # test_labels_vec = np.concatenate(test_labels_list, axis=0)
                test_predictions_vec = np.concatenate(test_predictions_list, axis=0)

                test_labels_vec = test_neighbor_avg_acc

            test_predictions_vec = test_predictions_vec[:len(test_labels_vec)]
            top_conf_inds = set(np.argwhere(test_labels_vec==1).flatten())
            bottom_conf_inds = set(np.argwhere(test_labels_vec==0).flatten())
            top_bias_inds = set(np.argwhere(test_predictions_vec==1).flatten())
            bottom_bias_inds = set(np.argwhere(test_predictions_vec==0).flatten())

            precision, recall, fpr, m11, m01 = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds)
            print('-'*100, len(top_conf_inds), len(bottom_conf_inds), len(top_bias_inds), len(bottom_bias_inds), '-'*100)

            n = test_labels_vec.shape[0]

            top_bias_inds_r = set(np.random.choice(n, len(top_bias_inds), replace=False))
            bottom_bias_inds_r = set([i for i in range(n)]) - set(top_bias_inds_r)

            precision_r, recall_r, fpr_r, m11_r, m01_r = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds_r, bottom_bias_inds_r)

            print('specs:', 'num_of_transformation_for_train', num_of_transformation_for_train, 'retrain_with_weak_points_for_detector', retrain_with_weak_points_for_detector, 'exp_num', exp_num)

            f1 = est_f1(precision, recall)
            f1_r = est_f1(precision_r, recall_r)
            print('ours & {:0.3f} & {:d} & {:d} & {:0.3f} & {:0.3f}'.format(f1, m11, m01, precision, recall))
            print('random & {:0.3f} & {:d} & {:d} & {:0.3f} & {:0.3f}'.format(f1_r, m11_r, m01_r, precision_r, recall_r))
        # Save the detector's prediction on the test set
        detector_test_result = {'test_labels_vec': test_labels_vec.squeeze(), 'test_predictions_vec': test_predictions_vec.squeeze()}
        np.savez(path_detector_test_result, **detector_test_result)

    # Get regression and binary classification results
    if running_mode in ['full', 'train_and_test', 'test', 'result'] and task_type == 'regression':
        d = np.load(path_detector_test_result)
        y = d['test_labels_vec']
        y_pred = d['test_predictions_vec'].squeeze()
        y = y[:y_pred.shape[0]]
        y_pred = y_pred[:y.shape[0]]
        print('regression error:', np.mean(np.abs(y-y_pred)))


        y_bin = bin(y, cutoff=test_cutoff).astype('int64')
        y_pred_bin = bin(y_pred, cutoff=test_cutoff).astype('int64')

        top_conf_inds = set(np.argwhere(y_bin==1).flatten())
        bottom_conf_inds = set(np.argwhere(y_bin==0).flatten())
        top_bias_inds = set(np.argwhere(y_pred_bin==1).flatten())
        bottom_bias_inds = set(np.argwhere(y_pred_bin==0).flatten())

        precision, recall, fpr, m11, m01 = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds)
        print('results:', precision, recall, fpr, m11, m01)
