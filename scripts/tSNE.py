# Set mode to 'two' for Figure 5 and '3d_tsne' for Figure 8
import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from train_detector import get_angle_threshold, estimate_neighbor_acc_by_threshold


def run_tSNE(natural_embed, n_jobs, perplexity):
    '''
    The GPU version requires CUDA 9.0 and install the tsnecuda package by running
    conda install tsnecuda -c cannylab

    The Multicore CPU version can be installed by running
    pip install MulticoreTSNE

    Apply t-SNE to the input data
    INPUT:
        natural_embed: 2d numpy array with size [number of points, embedding length]
        n_jobs:
        perplexity:
    OUTPUT:
        natural_2d: 2d numpy array with size [number of points, 2]
        adversary_2d: 2d numpy array with size [number of points, 2]
    '''
    X = natural_embed

    # CPU Sklearn
    # from sklearn.manifold import TSNE
    # tsne = TSNE(perplexity=perplexity, n_iter=5000, n_iter_without_progress=800, learning_rate=20, metric='cosine')
    # X_embedded = tsne.fit_transform(X)

    # CPU
    from MulticoreTSNE import MulticoreTSNE as TSNE
    tsne = TSNE(n_jobs=n_jobs, perplexity=perplexity, n_iter=5000, n_iter_without_progress=800, learning_rate=20, metric='cosine')
    X_embedded = tsne.fit_transform(X)

    # GPU
    # from tsnecuda import TSNE
    # X_embedded = TSNE(n_components=2, perplexity=30, learning_rate=10).fit_transform(X)

    return X_embedded


if __name__ == '__main__':
    if not os.path.exists('tsne'):
        os.mkdir('tsne')
    # Parse input parameters
    parser = argparse.ArgumentParser(description='Train an Image Classifier')
    parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--architecture', dest='architecture', type=str, default='resnet', help='architecture to use')

    args = parser.parse_args()
    architecture = args.architecture
    dataset = args.dataset

    assert architecture in ['resnet', 'wrn', 'vgg16', 'chauffeur', 'epoch', 'dave2']
    assert dataset in ['cifar10', 'svhn', 'fmnist', 'simulation']

    if dataset == 'simulation':
        transformation_mode = 'rainy_foggy_automold'
    else:
        transformation_mode = 'random_rotate_and_shift'

    path_to_only_center = 'tmp_data_without_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_'+transformation_mode+'_x4.npz'
    path_to_center_and_neighbors = 'tmp_data_with_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_'+transformation_mode+'_x4.npz'
    path_train_embed_and_neighbor_acc = 'tmp_data_without_neighbor/'+dataset+'/natural_train_embed_and_neighbor_acc_'+architecture+'_'+transformation_mode+'_50_x4.npz'

    n_jobs = 20
    perplexity = 30

    # ['full', 'after tsne']
    mode = 'full'
    # ['two', 'all', 'mon', '3d_tsne']
    # if 3d_tsne, other parameters will be ignored
    vis_mode = '3d_tsne'
    # activated when vis_mode == 'all'
    num_of_classes = 10
    sample_rate = 1


    path = path_to_only_center
    if vis_mode == 'mon':
        path = path_to_center_and_neighbors

    d = np.load(path)



    n = d['test_embed'].shape[0]
    th = int(n * sample_rate)

    test_embed = d['test_embed']
    test_y = d['test_y']
    test_y_pred = d['test_y_pred']

    if dataset == 'simulation':
        angle_threshold = get_angle_threshold(path_train_embed_and_neighbor_acc)
        test_neighbor_avg_acc = estimate_neighbor_acc_by_threshold(angle_threshold, test_y, test_y_pred)
    else:
        test_neighbor_avg_acc = d['test_neighbor_avg_acc'][:th]
        angle_threshold = 1
    test_embed = test_embed[:th]
    test_y = test_y[:th]
    test_y_pred = test_y_pred[:th]
    print(test_neighbor_avg_acc.shape)

    print('number of points :', n)

    if dataset == 'cifar10':
        objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'fmnist':
        objects = ['T-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    elif dataset == 'svhn':
        objects = [str(i) for i in range(10)]

    if vis_mode == '3d_tsne':

        remaining_idx = np.where((test_y - test_y_pred)**2 < angle_threshold)[0]
        print(np.mean(test_neighbor_avg_acc))
        print(len(remaining_idx))
        print(np.mean(test_neighbor_avg_acc[remaining_idx]))
        idx = remaining_idx
        print(np.mean(test_neighbor_avg_acc[idx]>0.75))
        x = run_tSNE(test_embed, n_jobs, perplexity)
        x = x[idx]


        fig, ax = plt.subplots()
        sc = plt.scatter(x[:, 0], x[:, 1], c=test_neighbor_avg_acc[idx], cmap=plt.cm.get_cmap('jet').reversed(), linewidth=0.1)
        plt.colorbar(sc)
        plt.savefig('tsne/'+dataset+'_'+architecture+'_sdc_.pdf')
        plt.show()


    elif vis_mode == 'two':
        i = 2
        j = 6
        fig, axs = plt.subplots(1, 1, figsize=(10,10), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        ax = axs

        if mode in ['full']:
            x1 = test_embed[test_y==i]
            x2 = test_embed[test_y==j]

            cutoff = x1.shape[0]
            x = run_tSNE(np.concatenate([x1, x2], axis=0), n_jobs, perplexity)
            x1 = x[:cutoff]
            x2 = x[cutoff:]

            np.savez('tsne/2d_'+dataset+'_'+architecture+'_'+str(i)+'_'+str(j), x1=x1, x2=x2)
            print('finish tSNE')

        if mode in ['full', 'after tsne']:
            d2 = np.load('tsne/2d_'+dataset+'_'+architecture+'_'+str(i)+'_'+str(j)+'.npz')
            x1, x2 = d2['x1'], d2['x2']

            c1 = test_neighbor_avg_acc[test_y==i]
            c2 = test_neighbor_avg_acc[test_y==j]

            ax.scatter(x1[:, 0], x1[:, 1], c=c1, s=50, cmap='Blues_r')
            ax.scatter(x2[:, 0], x2[:, 1], c=c2, s=50, cmap='Reds_r')
            print(x1.shape[0], x2.shape[0])
            print(c1)
            print(c2)

            h_natural = [plt.plot([],[], color=color, marker="o", ls="", ms=20)[0] for color in ['blue', 'red']]
            labels = [objects[i], objects[j]]

            legend_natural = ax.legend(handles=h_natural, labels=labels, loc=3, framealpha=0.3, prop={'size': 30})
            ax.add_artist(legend_natural)

            plt.show()

            fig.savefig('tsne/'+dataset+'_'+architecture+'_'+str(i)+'_'+str(j)+'.pdf', dpi=1000, bbox_inches = 'tight')
            plt.close(fig)

    elif vis_mode == 'all':
        fig, axs = plt.subplots(1, 1, figsize=(10,10), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        ax = axs

        if mode in ['full']:
            x_all = run_tSNE(test_embed, n_jobs, perplexity)
            np.savez('tsne/'+architecture+'_2d_all', x_all=x_all)
            print('finish tSNE')

        if mode in ['full', 'after tsne']:
            d2 = np.load('tsne/'+architecture+'_2d_all.npz')
            x_all = d2['x_all']


            # colors = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'YlOrBr_r', 'YlOrRd_r', 'OrRd_r', 'PuRd_r']
            #
            # for i in range(num_of_classes):
            #     xi = x_all[test_y==i]
            #     ci = test_neighbor_avg_acc[test_y==i]
            #     ax.scatter(xi[:, 0], xi[:, 1], c=ci, s=10, cmap=colors[i])
            #     print(i, xi.shape[0])
            # plt.show()

            # Two colors
            low_acc_colors = ['blue', 'black', 'green', 'red', 'purple', 'yellow', 'deeppink', 'gray', 'royalblue', 'cyan']

            high_acc_colors = ['lightblue', 'dimgray', 'lightgreen', 'lightcoral', 'violet', 'lightyellow', 'hotpink', 'lightgray', 'cornflowerblue', 'lightcyan']


            threshold = 0.5
            for i in range(num_of_classes):
                xi = None

                for j in range(2):
                    if j == 1:
                        color = low_acc_colors[i]
                        xi = x_all[(test_y==i) & (test_neighbor_avg_acc<threshold)]
                    else:
                        color = high_acc_colors[i]
                        xi = x_all[(test_y==i) & (test_neighbor_avg_acc>=threshold)]

                    ax.scatter(xi[:, 0], xi[:, 1], c=color, s=15)
                    print(i, xi.shape[0])

            h_natural = [plt.plot([],[], color=low_acc_colors[i], marker="o", ls="", ms=10)[0] for i in range(10)]
            labels = [objects[i] for i in range(10)]
            legend_natural = ax.legend(handles=h_natural, labels=labels, loc=1, framealpha=0.0, prop={'size': 15})
            legend_natural.set_title('weak points', prop={'size':15})
            ax.add_artist(legend_natural)

            h_natural = [plt.plot([],[], color=high_acc_colors[i], marker="o", ls="", ms=10)[0] for i in range(10)]
            labels = [objects[i] for i in range(10)]
            legend_natural = ax.legend(handles=h_natural, title='strong points', labels=labels, loc=2, framealpha=0.0, prop={'size': 15})
            legend_natural.set_title('strong points', prop={'size':15})
            ax.add_artist(legend_natural)

            plt.show()


        fig.savefig('tsne/'+dataset+'_'+architecture+'_all_bi'+'.pdf', dpi=1000, bbox_inches = 'tight')
        plt.close(fig)

    elif vis_mode == 'mon':
        num_test_nat_batches = 200
        batch_size = 50
        num_of_transformation_for_test = 50
        acc_th = 0.5

        example_num = num_test_nat_batches * batch_size

        center_embed = test_embed[:example_num]
        center_y = test_y[:example_num]
        center_y_pred = test_y_pred[:example_num]
        center_acc = test_neighbor_avg_acc[:example_num]

        t = np.zeros(example_num)
        for ii in range(example_num):
            correct_num = 0
            begin_wrong_left = False
            begin_wrong_right = False
            is_mon = True

            # cor_ii_list = []
            # cor_jj_list = []

            num_of_transformation_for_test_half = num_of_transformation_for_test // 2

            for jj in range(1, num_of_transformation_for_test_half+1):
                cur_ind = ii + jj * example_num

                y = test_y[cur_ind]
                y_pred = test_y_pred[cur_ind]

                if y == y_pred:
                    if begin_wrong_left:
                        is_mon = False
                        break
                    # cor_ii_list.append(0)

                else:
                    begin_wrong_left = True
                    # cor_ii_list.append(1)

            # if is_mon:
            for jj in range(num_of_transformation_for_test_half+1, num_of_transformation_for_test+1):
                cur_ind = ii + jj * example_num

                y = test_y[cur_ind]
                y_pred = test_y_pred[cur_ind]

                if y == y_pred:
                    if begin_wrong_right:
                        is_mon = False
                        break
                    # cor_jj_list.append(0)
                else:
                    begin_wrong_right = True
                    # cor_jj_list.append(1)

            if center_acc[ii] < acc_th:
                t[ii] = 0
            elif is_mon:
                t[ii] = 1
            else:
                t[ii] = 2
            # print(cor_ii_list, cor_jj_list)



        # Choose one class
        ind = 8
        t = t[center_y == ind]

        fig, axs = plt.subplots(1, 1, figsize=(10,10), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        ax = axs

        if mode in ['full']:
            x1 = center_embed[center_y==ind]
            print(x1.shape, 'for running tSNE')

            x = run_tSNE(x1, n_jobs, perplexity)
            x1 = x

            np.savez('tsne/2d_'+vis_mode+'_'+str(ind), x1=x1)
            print('finish tSNE')

        if mode in ['full', 'after tsne']:
            d2 = np.load('tsne/2d_'+vis_mode+'_'+str(ind)+'.npz')
            x1 = d2['x1']

            # low acc
            x1_0 = x1[t==0]

            # mono
            x1_1 = x1[t==1]

            # non-mono
            x1_2 = x1[t==2]

            ax.scatter(x1_0[:, 0], x1_0[:, 1], c='grey', s=50)
            ax.scatter(x1_1[:, 0], x1_1[:, 1], c='blue', s=50)
            ax.scatter(x1_2[:, 0], x1_2[:, 1], c='red', s=50)
            print(x1_0.shape[0], x1_1.shape[0], x1_2.shape[0])
            print('neighbor acc :', np.mean(center_acc[center_y == ind][t==0]), np.mean(center_acc[center_y == ind][t==1]), np.mean(center_acc[center_y == ind][t==2]))
            print('avg neighbor acc :', np.mean(center_acc))

            plt.show()


        fig.savefig('tsne/'+vis_mode+'_'+str(ind)+'.pdf', dpi=1000, bbox_inches = 'tight')
        plt.close(fig)
