'''
tf.data for loading CIFAR10
train an old model for CIFAR10

The code should be extensible to other datasets without too much modification

'''
import json
import os
import shutil
import socket
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

import dataloader.loader


# from learning.model_mnist import ModelMNIST, ModelMNISTBN
# from learning.model_imagenet_wrn_small import ModelImagenet


from utils_new import trainable_in, remove_duplicate_node_from_list, visualize_imgs, include_patterns

import matplotlib.pyplot as plt




# Parse input parameters
parser = argparse.ArgumentParser(description='Train an Image Classifier')
parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--architecture', dest='architecture', type=str, default='resnet', help='architecture to use')

args = parser.parse_args()
architecture = args.architecture
dataset_type = args.dataset


# ['train', 'test']
mode = 'train'
# ['plain', 'augmentation', 'rot_and_shift', 'rot_and_shift_exact']
train_input_mode = 'augmentation'
train_angle = 0
train_x_shift = 2
train_y_shift = 2
test_input_mode = 'plain'
test_angle = 0
test_x_shift = 0
test_y_shift = 0
retrain_with_weak_points = False
# Temporarily set is_finetune here rather than in the config file
is_finetune = False

flip = False
if dataset_type in ['cifar10']:
    flip = True
    train_x_shift = 2
    train_y_shift = 2
    # train_input_mode = 'rot_and_shift'
    # train_angle = 10
    # test_x_shift = 2
    # test_y_shift = 2
else:
    train_input_mode = 'plain'
    flip = False
    train_x_shift = 0
    train_y_shift = 0

assert architecture in ['resnet', 'wrn', 'vgg16']
assert dataset_type in ['cifar10', 'svhn', 'fmnist', 'celeb5', 'celebA']

if architecture == 'resnet':
    from learning.model_resnet import ModelVani
elif architecture == 'wrn':
    from learning.model_wrn import ModelVani
elif architecture == 'vgg16':
    from learning.model_vgg16 import ModelVani
else:
    raise

if dataset_type in ['cifar10', 'svhn', 'celeb5', 'celebA']:
    input_shape = [None, 32, 32, 3]
elif dataset_type == 'fmnist':
    input_shape = [None, 28, 28, 1]
else:
    raise


# Load in config files and set up parameters

with open('configs/config_'+dataset_type+'.json') as config_file:
    config = json.load(config_file)


max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
weight_decay = config['weight_decay']
momentum = config['momentum']
batch_size = config['training_batch_size']
optimizer = config['optimizer']
# Path to save the model
model_dir = config['model_dir_ziyuan']

# Path to data
data_path = config['data_path']

retrain_with_weak_points_str = ''
if retrain_with_weak_points:
    retrain_with_weak_points_str = '_mask'
finetune_str = ''
if is_finetune:
    finetune_str = '_finetune'

# Path to load model
model_load_dir = config['model_load_dir_ziyuan']+'/natural'+'_'+architecture


step_size_schedule = config['step_size_schedule']

if is_finetune:
    maximum_epochs = 20
    step_size_schedule = config['step_size_schedule_finetune']
    # 1 epoch for each dataset
    if dataset_type == 'cifar10':
        max_num_training_steps = int(50000 / batch_size) * maximum_epochs
    elif dataset_type == 'fmnist':
        max_num_training_steps = int(60000 / batch_size) * maximum_epochs
    elif dataset_type == 'svhn':
        max_num_training_steps = int(73257 / batch_size) * maximum_epochs
    elif dataset_type == 'celeb5':
        max_num_training_steps = 25*20
    elif dataset_type == 'celebA':
        max_num_training_steps = 25*20
    else:
        raise
    train_input_mode = 'rot_and_shift'
    train_angle = 30
    train_x_shift = 3
    train_y_shift = 3


if dataset_type in ['cifar10', 'cifar100']:
    num_testing_steps = int(np.ceil(10000 / batch_size))
elif dataset_type in ['fmnist']:
    num_testing_steps = int(np.ceil(10000 / batch_size))
elif dataset_type in ['svhn']:
    num_testing_steps = int(np.ceil(26032 / batch_size))



gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='1')

# Change path according to host
if socket.gethostname() == 'deep':
    model_dir = config['model_dir']
    model_load_dir = config['model_load_dir']

tmp_data_folder = 'tmp_data_without_neighbor/'+dataset_type
model = 'natural'
using_neighbors_for_detection_str = ''
vis_mode_str = ''
layer_str = '_x4'
train_transformation_mode_str = '_sweeping_rotate_and_random_shift'
path_train_embed_and_neighbor_acc = tmp_data_folder+'/'+model+'_'+'train_embed_and_neighbor_acc'+using_neighbors_for_detection_str+'_'+architecture+train_transformation_mode_str+vis_mode_str+layer_str+'.npz'

model_dir += model + '_'+ architecture + finetune_str + retrain_with_weak_points_str

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

global_step = tf.train.get_or_create_global_step()


sample_prob = None
n = 0
if retrain_with_weak_points:
    # Add probability with smoothing technique
    d = np.load(path_train_embed_and_neighbor_acc)
    sample_prob = (d['train_neighbor_avg_acc'])*100
    sample_prob = (np.ones_like(sample_prob)+sample_prob) / 101
    # Use square to exaggerate low neighbor acc
    # sample_prob = sample_prob**2
    sample_prob = sample_prob[:, np.newaxis]

    n = sample_prob.shape[0]
    slice_number = n // batch_size
    if n % batch_size != 0:
        slice_number += 1



# Initialize Model and Dataset Loader
model = ModelVani(dataset_type, retrain_with_weak_points)
dataload = dataloader.loader.build_loader(data_path, dataset_type)


with tf.variable_scope('input'):
    images = tf.placeholder(tf.float32, shape=input_shape)
    labels = tf.placeholder(tf.int64, shape=None)
    is_training = tf.placeholder(tf.bool, shape=None)
    sample_prob_tf = tf.placeholder(tf.float32, shape=[batch_size, 1])

# TBD: graph for the test mode
images_transformed = dataloader.loader.transform(images, mode=train_input_mode, angle=train_angle, x_shift=train_x_shift, y_shift=train_y_shift, flip=flip)
layer_values, n_Amean_xent, n_Aweight_decay_loss, _, n_Aaccuracy, pred, _ = \
    model._encoder(images_transformed, labels, is_training, sample_prob_tf)
lv_x3 = layer_values['x3']

print("finish building up model")

# Restore trained model
var_main_encoder = trainable_in('main_encoder')
var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
saver_restore = tf.train.Saver(restore_var_list)


### Caculate losses
with tf.variable_scope('train/m_encoder_momentum'):
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    if dataset_type == 'celebA' and architecture == 'vgg16':
        values = [v * 0.1 for v in values]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)

    total_loss = n_Amean_xent
    total_loss += weight_decay * n_Aweight_decay_loss

    encoder_opt = None
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

tf.summary.scalar('train_batch_nat accuracy', n_Aaccuracy)
tf.summary.scalar('train_batch_nat mean xent', n_Amean_xent)
tf.summary.scalar('lr', learning_rate)
merged_summaries = tf.summary.merge_all()

# To avoid folder name conflict, append index in the end.
new_model_dir = model_dir
postfix_ind = 0
while os.path.exists(new_model_dir):
    new_model_dir = model_dir  + '_' + str(postfix_ind)
    postfix_ind += 1

if mode == 'train':
    # Save all the code and config for the saved model
    shutil.copytree('.', os.path.join(new_model_dir), ignore=include_patterns('*.py', '*.json'))
model_dir = new_model_dir

fp = open(os.path.join(model_dir, 'log.txt'), 'a')



if mode == 'train':
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        eval_dir = os.path.join(model_dir, 'eval')
        test_summary_writer = tf.summary.FileWriter(eval_dir+'_nat')
        sess.run(tf.global_variables_initializer())

        # Restore trained model
        if is_finetune:
            model_dir_load = tf.train.latest_checkpoint(model_load_dir)
            saver_restore.restore(sess, model_dir_load)

        best_acc = 0

        if retrain_with_weak_points:
            train_loaded_data = np.load(path_train_embed_and_neighbor_acc)
            train_neighbor_avg_acc = train_loaded_data['train_neighbor_avg_acc']

        for ii in range(max_num_training_steps):
            # x_batch, y_batch = dataload.train_data.get_next_batch(batch_size, multiple_passes=True)
            x_batch, y_batch = None, None
            if retrain_with_weak_points:
                x_batch, y_batch = dataload.train_data.get_next_batch_reweight(batch_size, train_neighbor_avg_acc)
            else:
                x_batch, y_batch = dataload.train_data.get_next_batch(batch_size, reshuffle_after_pass=True)

            # print(x_batch[0], y_batch[0])
            # plt.imshow(x_batch[0])
            # plt.show()


            sample_prob_slice = np.ones([batch_size, 1])
            if retrain_with_weak_points:
                ind = ii % slice_number
                left = ind * batch_size
                right = np.min([n, (ind+1) * batch_size])
                sample_prob_slice = sample_prob[left:right]

                if (right - left) < batch_size:
                    pad_size = batch_size - (right - left)
                    padding = np.zeros([pad_size, 1])
                    sample_prob_slice = np.concatenate([sample_prob_slice, padding], axis=0)
            else:
                sample_prob_slice = np.ones([batch_size, 1])
            sample_prob_slice = sample_prob_slice*batch_size/np.sum(sample_prob_slice)

            sample_prob_slice_constant = np.ones([batch_size, 1])

            # TBD: this is temporary for uniform mask
            sample_prob_slice = np.ones([batch_size, 1])



            # print(sample_prob_slice[:5])
            _ = sess.run(train_step_m_encoder, feed_dict={images: x_batch.astype(np.float32),
                              labels: y_batch.astype(np.float32), is_training: True, sample_prob_tf: sample_prob_slice})

            if ii % num_output_steps == 0:

                images_vis, nat_acc, nat_xent_value = sess.run([images, n_Aaccuracy, n_Amean_xent], feed_dict={images: x_batch.astype(np.float32),
                                  labels: y_batch.astype(np.float32), is_training: False, sample_prob_tf: sample_prob_slice_constant})
                str1 = 'Step {}:    ({})\n'.format(ii, datetime.now()) \
                       + 'training nat batch accuracy {:.4}%\n'.format(nat_acc * 100) \
                       + 'training nat xent {:.4}%\n'.format(nat_xent_value)
                print(str1)

            if ii % num_summary_steps == 0:
                summary = sess.run(merged_summaries, feed_dict={images: x_batch.astype(np.float32),
                                  labels: y_batch, is_training: False, sample_prob_tf: sample_prob_slice_constant})
                test_summary_writer.add_summary(summary, global_step.eval(sess))

            if ii % num_checkpoint_steps == 0 and ii > 0:

                nat_acc_list = []
                for ii in range(num_testing_steps):
                    x_batch_test, y_batch_test = dataload.eval_data.get_next_batch(batch_size, multiple_passes=True)
                    nat_acc = sess.run(n_Aaccuracy, feed_dict={images: x_batch_test.astype(np.float32),
                                      labels: y_batch_test, is_training: False, sample_prob_tf: sample_prob_slice_constant})
                    nat_acc_list.append(nat_acc)

                test_acc = np.mean(nat_acc_list)
                print('\n********')
                print('cur test accuracy {:.4}%:'.format(test_acc * 100))
                print('\n********')
                if test_acc > best_acc:
                    saver_restore.save(sess, os.path.join(model_dir, 'checkpoint_{}'.format(str(test_acc))), global_step=global_step)
                    best_acc = test_acc

elif mode == 'test':
    sample_prob_slice_constant = np.ones([batch_size, 1])
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # Restore trained model
        model_dir_load = tf.train.latest_checkpoint(model_load_dir)
        saver_restore.restore(sess, model_dir_load)

        nat_acc_list = []
        for ii in range(num_testing_steps):
            x_batch_test, y_batch_test = dataload.eval_data.get_next_batch(batch_size, multiple_passes=False)

            nat_acc = sess.run([n_Aaccuracy], feed_dict={images: x_batch_test.astype(np.float32),
                              labels: y_batch_test, is_training: False, sample_prob_tf: sample_prob_slice_constant})


            # str1 = 'Step {}:    ({})\n'.format(ii, datetime.now()) \
            #        + 'testing nat batch accuracy {:.4}%\n'.format(nat_acc * 100)
            # print(str1)
            nat_acc_list.extend(nat_acc*y_batch_test.shape[0])

        print('average testing accuracy :', np.mean(nat_acc_list))
