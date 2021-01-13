import argparse
import logging
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import model_factory
import utils_train_self_driving_car
from batch_generator import Generator

from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

import sklearn
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
np.random.seed(0)
logger = logging.Logger("train_self_driving_car")


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint(args.model + '-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    train_generator = Generator(X_train, y_train, True, args)
    validation_generator = Generator(X_valid, y_valid, False, args)

    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        epochs=args.nb_epoch,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        workers=4,
                        callbacks=[checkpoint],
                        verbose=1)



def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def get_embed_helper(model, args, X, y, mode, layer_name, dtheta=0):
    intermediate_layer_model = Model(inputs=model.input,
                             outputs=(model.get_layer(layer_name).output, model.output))
    validation_generator = Generator(X, y, False, args, mode, dtheta)
    intermediate_output = intermediate_layer_model.predict_generator(validation_generator, callbacks=None, verbose=1)

    embed = intermediate_output[0]
    y_pred = intermediate_output[1].squeeze()

    # discard the last few examples
    l1_dist = np.abs(y_pred - y[:len(y_pred)])
    import sklearn
    mse = sklearn.metrics.mean_squared_error(y[:len(y_pred)], y_pred)

    # print(mode, 'mean error :', np.mean(l1_dist), '% > 0.1', np.mean(l1_dist>0.1))
    return embed, y_pred, l1_dist

def train():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='../datasets/dataset5/')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=500)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=256)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-sl', help='sequence length', dest='sequence_length', type=int, default=3)
    parser.add_argument('-m', help='model', dest='model', type=str, default="chauffeur")
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # TBD: change to relative path
    args.data_dir = '/home/zhongzzy9/Documents/self-driving car/misbehavior_prediction/datasets/dataset5'
    data = utils_train_self_driving_car.load_data(args)
    model = model_factory.get_model(args)
    assert model is not None
    train_model(model, args, *data)

def get_embed(model, dataset, neighbor_num, transformation_mode='random_rotate_and_shift'):

    checkpoint_path = '/home/zhongzzy9/Documents/self-driving car/misbehavior_prediction/trained_sdc'

    if args.model == 'chauffeur':
        checkpoint_path += '/chauffeur-315.h5'
        layer_name = 'flatten_1'
    elif args.model == 'dave2':
        checkpoint_path += '/dave2-dataset5-823.h5'
        layer_name = 'dense_3'
    elif args.model == 'epoch':
        checkpoint_path += '/epoch-dataset5-304.h5'
        layer_name = 'dense_1'
    model.load_weights(checkpoint_path)

    X_train, X_valid, y_train, y_valid = data

    if dataset == 'train':
        X, y = X_train, y_train
    elif dataset == 'test':
        X, y = X_valid, y_valid


    # # following DeepTest
    # eps = 0.03


    y_pred_list = []
    l1_dist_list = []

    embed, y_pred_0, l1_dist_0 = get_embed_helper(model, args, X, y, 'plain', layer_name, 0)

    mse_ori = np.mean(l1_dist_0)
    y_pred_list.append(y_pred_0)
    l1_dist_list.append(l1_dist_0[:, np.newaxis])

    valid_angles = []

    if transformation_mode == 'exact_rotate':
        if dataset == 'train':
            angles = [i for i in range(1, 10)] + [-i for i in range(1, 10)]
        else:
            assert os.path.exists('valid_angles.npy')
            angles = np.load('valid_angles.npy').tolist()

        for d_theta in angles:
            print(d_theta)
            _, y_pred_i, l1_dist_i = get_embed_helper(model, args, X, y, transformation_mode, layer_name, d_theta)

            mse_i = np.mean(l1_dist_i)
            print('mse_ori', mse_ori, 'mse_i', mse_i)
            if dataset == 'train' and np.abs(mse_i-mse_ori) < eps:
                y_pred_list.append(y_pred_i)
                l1_dist_list.append(l1_dist_i[:, np.newaxis])
                valid_angles.append(d_theta)
            elif dataset == 'test':
                y_pred_list.append(y_pred_i)
                l1_dist_list.append(l1_dist_i[:, np.newaxis])
    else:
        for i in range(neighbor_num):
            print(i)
            _, y_pred_i, l1_dist_i = get_embed_helper(model, args, X, y, transformation_mode, layer_name)
            y_pred_list.append(y_pred_i)
            l1_dist_list.append(l1_dist_i[:, np.newaxis])
            mse_i = np.mean(l1_dist_i)
            print('mse_ori', mse_ori, 'mse_i', mse_i)
    if dataset == 'train' and transformation_mode == 'exact_rotate':
        valid_angles = np.array(valid_angles)
        np.save('valid_angles', valid_angles)

    l1_dist = np.concatenate(l1_dist_list, axis=1).squeeze()
    mean_l1_dist = np.mean(l1_dist, axis=1)

    y_pred = np.concatenate(y_pred_list)

    return embed, y[:len(y_pred_0)], y_pred, mean_l1_dist



if __name__ == '__main__':

    # ['train', 'embed_train_and_test']
    running_mode = 'embed_train_and_test'
    neighbor_num = 50
    folder = 'tmp_data_without_neighbor/simulation/'
    if not os.path.exists(folder):
        os.mkdir(folder)

    if running_mode == 'embed_train_and_test':
        parser = argparse.ArgumentParser(description='Behavioral Cloning Testing Program')
        parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='../datasets/dataset5/')
        parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
        parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
        parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=64)
        parser.add_argument('-sl', help='sequence length', dest='sequence_length', type=int, default=3)
        parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
        parser.add_argument('-m', help='model', dest='model', type=str, default="chauffeur")
        args = parser.parse_args()

        # 'rainy_foggy_automold', 'rainy_foggy_iaa', 'exact_rotate'
        transformation_mode = 'rainy_foggy_automold'
        path_train_embed_and_neighbor_acc = folder + 'natural_train_embed_and_neighbor_acc_'+args.model+'_'+transformation_mode+'_'+str(neighbor_num)+'_x4'
        path_test_embed_and_neighbor_acc = folder + 'natural_test_embed_and_neighbor_acc_'+args.model+'_'+transformation_mode+'_x4'
        args.data_dir = '/home/zhongzzy9/Documents/self-driving car/misbehavior_prediction/datasets/dataset5'
        data = utils_train_self_driving_car.load_data(args)
        model = model_factory.get_model(args)
        assert model is not None
        print(model.summary())

        embed_train, y_train, y_pred_train, mean_l1_dist_train = get_embed(model, 'train', neighbor_num, transformation_mode)
        transformed_embed_train = embed_train
        # if model in ['epoch', 'chauffeur']:
        #     from sklearn.decomposition import PCA
        #     pca = PCA(n_components=300)
        #     pca.fit(embed_train)
        #     print('explained_variance_ratio_',np.sum(pca.explained_variance_ratio_))
        #     print(embed_train.shape)
        #     transformed_embed_train = pca.transform(embed_train)
        #     print(transformed_embed_train.shape)
        # else:
        #     transformed_embed_train = embed_train

        train_saved_data = {'train_embed': transformed_embed_train, 'train_y': y_train, 'train_y_pred': y_pred_train, 'train_neighbor_avg_acc': mean_l1_dist_train}
        np.savez(path_train_embed_and_neighbor_acc, **train_saved_data)


        embed_test, y_test, y_pred_test, mean_l1_dist_test = get_embed(model, 'test', neighbor_num, transformation_mode)
        transformed_embed_test = embed_test
        # if model in ['epoch', 'chauffeur']:
        #     transformed_embed_test = pca.transform(embed_test)
        # else:
        #     transformed_embed_test = embed_test

        test_saved_data = {'test_embed': transformed_embed_test, 'test_y': y_test, 'test_y_pred': y_pred_test, 'test_neighbor_avg_acc': mean_l1_dist_test}
        np.savez(path_test_embed_and_neighbor_acc, **test_saved_data)

    elif running_mode == 'train':
        train()
