'''
Set task to
'avg_ratio_w/s_d_to_centroid'for Table 2,
'spearmanr_cor' for Table 3,
'spearmanr_si_neighbor_acc_wrt_neighbor_num' for Figure 7,
'pred_using_top_1' for top1 baseline in Table 4 and Table 6,
'pred_using_si' for our method in Table 6
'''


import numpy as np
import itertools
import os
from scipy.stats import spearmanr, mannwhitneyu

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from utils_new import get_single_precision_recall_fpr, get_single_precision_recall_fpr_helper, est_f1


import matplotlib.pyplot as plt
from train_detector import bin
# ['spearmanr_si_neighbor_acc_wrt_neighbor_num', 'spearmanr_cor', 'avg_ratio_w/s_d_to_centroid', 'prec_recall',
# 'prec_recall_changing_cutoff_y_pred', 'diff_training_mode',
# 'diff_neighbor_num', 'pred_using_si', 'pred_using_top_1', 'close_to_own_ratio', 'find_one_weak_diverse_region',
# 'roc_using_si_and_top1']

def get_si(d, example_num, mode_str, num_of_transformation=None):

    y_pred = d[mode_str+'_y_pred']

    num_of_classes = 10
    if not num_of_transformation:
        num_of_transformation = 50

    si_arr = np.zeros(example_num)

    for ii in range(example_num):
        ii_y_pred_counter = np.zeros(num_of_classes)

        for jj in range(0, num_of_transformation+1):
            cur_ind = ii + jj * example_num
            y_pred_ind = y_pred[cur_ind]
            ii_y_pred_counter[y_pred_ind] += 1
        si = np.sum(ii_y_pred_counter **2) / (num_of_transformation+1)**2
        si_arr[ii] = si
    return si_arr

task = 'spearmanr_si_neighbor_acc_wrt_neighbor_num'

if task == 'roc_using_si_and_top1':
    roc_data = dict()
if task == 'spearmanr_si_neighbor_acc_wrt_neighbor_num':
    spear = dict()

# ['cifar10', 'svhn', 'fmnist', 'celeb5', ’celebA]
datasets = ['cifar10', 'svhn', 'fmnist']
# ['resnet', 'wrn', 'vgg16']
architectures = ['resnet', 'wrn', 'vgg16']
# ['random_rotate_and_shift', 'sweeping_rotate_and_shift', 'sweeping_rotate_and_random_shift', 'sweeping_rotate_and_25_random_shift']
# train_transformation_modes = ['random_rotate_and_shift', 'sweeping_rotate_and_shift', 'sweeping_rotate_and_random_shift', 'sweeping_rotate_and_25_random_shift']
train_transformation_modes = ['random_rotate_and_shift']
nums_of_transformation_for_train = [50]


settings = list(itertools.product(datasets, architectures, train_transformation_modes, nums_of_transformation_for_train))

retrain_with_weak_points_for_detector = False
exp_num = 0
retrain_with_weak_points_for_detector_str = ''
if retrain_with_weak_points_for_detector:
    retrain_with_weak_points_for_detector_str = '_detector_mask'+'_'+str(exp_num)

for dataset, architecture, train_transformation_mode, num_of_transformation_for_train in settings:
    transformation_mode_str = '_'+train_transformation_mode
    if train_transformation_mode == 'random_rotate_and_shift':
        transformation_mode_str += ('_'+str(num_of_transformation_for_train))

    setting_name = architecture+'_'+dataset+transformation_mode_str
    print(setting_name)

    if task == 'spearmanr_cor':
        path_test_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_x4.npz'
        d = np.load(path_test_embed_and_neighbor_acc)



        example_num = 0
        if dataset in ['cifar10', 'fmnist']:
            example_num = 10000
        elif dataset in ['svhn']:
            example_num = 26032


        center_y = d['test_y'][:example_num]
        center_y_pred = d['test_y_pred'][:example_num]
        center_acc = d['test_neighbor_avg_acc'][:example_num]
        test_y_pred = d['test_y_pred']

        num_of_classes = 10
        num_of_transformation_for_test = 50
        # Due to the memory issue casued by high dimension of vgg16 x4 layer
        if architecture == 'vgg16' and dataset == 'svhn':
            num_of_transformation_for_test = 50


        si_arr = np.zeros(example_num)
        for ii in range(example_num):
            ii_y_pred_counter = np.zeros(num_of_classes)

            for jj in range(1, num_of_transformation_for_test+1):
                cur_ind = ii + jj * example_num
                y_pred = test_y_pred[cur_ind]
                ii_y_pred_counter[y_pred] += 1
            si = np.sum(ii_y_pred_counter **2) / (num_of_transformation_for_test**2)
            si_arr[ii] = si

        center_acc_sub = center_acc[center_acc < 1.1]
        si_arr_sub = si_arr[center_acc < 1.1]
        print(np.mean(center_y==center_y_pred), np.mean(d['test_y']==d['test_y_pred']), spearmanr(center_acc_sub, si_arr_sub))

    if task == 'spearmanr_si_neighbor_acc_wrt_neighbor_num':
        path_test_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_x4.npz'
        d_test = np.load(path_test_embed_and_neighbor_acc)

        test_example_num = 0
        if dataset in ['cifar10']:
            test_example_num = 10000
        elif dataset in ['fmnist']:
            test_example_num = 10000
        elif dataset in ['svhn']:
            test_example_num = 26032

        test_center_acc = d_test['test_neighbor_avg_acc'][:test_example_num]
        spear_list = []
        for si_num in range(2, 51, 3):
            test_si = get_si(d_test, test_example_num, 'test', si_num)
            spear_list.append(spearmanr(test_si, test_center_acc)[0])
        print(test_si[75], test_center_acc[75])
        print(test_si[674], test_center_acc[674])
        spear[dataset+architecture] = spear_list


    if task in ['pred_using_si', 'roc_using_si_and_top1']:
        path_test_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_x4.npz'
        d_test = np.load(path_test_embed_and_neighbor_acc)
        path_train_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_train_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_50_x4.npz'
        d_train = np.load(path_train_embed_and_neighbor_acc)

        train_example_num = 0
        test_example_num = 0
        if dataset in ['cifar10']:
            train_example_num = 50000
            test_example_num = 10000
        elif dataset in ['fmnist']:
            train_example_num = 60000
            test_example_num = 10000
        elif dataset in ['svhn']:
            train_example_num = 73257
            test_example_num = 26032

        test_center_acc = d_test['test_neighbor_avg_acc'][:test_example_num]
        train_center_acc = d_train['train_neighbor_avg_acc'][:train_example_num]

        num_for_si = 50
        train_si = get_si(d_train, train_example_num, 'train', num_for_si)
        test_si = get_si(d_test, test_example_num, 'test', num_for_si)


        train_center_acc_cutoff = 0.75
        test_center_acc_cutoff = train_center_acc_cutoff

        from scipy import stats
        train_perc = stats.percentileofscore(train_center_acc, train_center_acc_cutoff)/100
        # test_perc = stats.percentileofscore(test_center_acc, test_center_acc_cutoff)/100

        train_si_cutoff = np.quantile(train_si, train_perc)

        print('si_cutoff', train_si_cutoff)
        test_si_cutoff = train_si_cutoff

        # test_random = np.random.random(test_si.shape)
        # ap = average_precision_score(test_center_acc_bi, 1-test_si)
        # ap_random = average_precision_score(test_center_acc_bi, test_random)
        # print('cutoff:', train_center_acc_cutoff, 'ap: {:0.4f} ap_random: {:0.4f}'.format(ap, ap_random))
        #
        top_conf_inds = set(np.argwhere(test_center_acc<=test_center_acc_cutoff).flatten())
        bottom_conf_inds = set(np.argwhere(test_center_acc>test_center_acc_cutoff).flatten())



        top_bias_inds = set(np.argwhere(test_si<=test_si_cutoff).flatten())
        bottom_bias_inds = set(np.argwhere(test_si>test_si_cutoff).flatten())

        precision, recall, fpr, m11, m01 = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds)

        top_bias_inds_r = set(np.random.choice(test_example_num, len(top_bias_inds), replace=False))
        bottom_bias_inds_r = set([i for i in range(test_example_num)]) - set(top_bias_inds_r)

        precision_r, recall_r, fpr_r, m11_r, m01_r = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds_r, bottom_bias_inds_r)
        f1 = est_f1(precision, recall)
        f1_r = est_f1(precision_r, recall_r)
        print('ours & {:0.3f} & {:0.3f} & {:d} & {:d} \\\\'.format(f1, m11, m01))
        print('random & {:0.3f} & {:0.3f} & {:d} & {:d} \\\\'.format(f1_r, m11_r, m01_r))

        # from sklearn.svm import SVR
        # from sklearn.linear_model import LinearRegression
        # #regressor = SVR(kernel='linear', C=1e3)
        # regressor = LinearRegression()
        #
        # from sklearn.preprocessing import PolynomialFeatures
        #
        #
        # #Edit: added second square bracket above to fix the ValueError problem
        #
        # poly = PolynomialFeatures(degree=2)
        # X_ = poly.fit_transform(train_si.reshape(-1, 1))
        # predict_ = poly.fit_transform(test_si.reshape(-1, 1))
        #
        # regressor = LinearRegression()
        # regressor = regressor.fit(X_, train_center_acc)
        # test_center_acc_pred = regressor.predict(predict_)
        #
        # #print(train_center_acc[:20], test_center_acc[:20])
        # # regressor = regressor.fit(train_si.reshape(-1, 1), train_center_acc)
        # # test_center_acc_pred = regressor.predict(test_si.reshape(-1, 1))
        #
        # #print(test_center_acc_pred[:20], test_center_acc[:20])
        #
        # err = test_center_acc_pred[test_center_acc_pred<0.5] - test_center_acc[test_center_acc_pred<0.5]
        # err_mean = np.mean(err)
        # err_std = np.std(err)
        #
        # print('{:0.3f} {:0.3f} {:0.3f}'.format(err_mean, err_std, np.percentile(np.abs(err), 90)))
        # plt.hist(err)
        # plt.axvline(np.percentile(np.abs(err), 90), color='k', linestyle='dashed', linewidth=1)
        # plt.axvline(-np.percentile(np.abs(err), 90), color='k', linestyle='dashed', linewidth=1)
        #
        # plt.savefig('tmp_img/err_dist_'+dataset+'_'+architecture)
        # # plt.scatter(test_si, test_center_acc)
        # # plt.show()
        #
        #
        #
        # import matplotlib.cm as cm
        # from scipy.ndimage.filters import gaussian_filter
        #
        #
        # test_si = test_si[test_center_acc<0.9]
        # test_center_acc = test_center_acc[test_center_acc<0.9]
        #
        #
        # def myplot(x, y, s, bins=1000):
        #     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        #     heatmap = gaussian_filter(heatmap, sigma=s)
        #
        #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #     return heatmap.T, extent
        #
        #
        # fig, axs = plt.subplots(2, 2)
        #
        # # Generate some test data
        # x = test_si
        # y = test_center_acc
        #
        # sigmas = [0, 16, 32, 64]
        #
        # for ax, s in zip(axs.flatten(), sigmas):
        #     if s == 0:
        #         ax.plot(x, y, 'k.', markersize=5)
        #         ax.set_title("Scatter plot")
        #     else:
        #         img, extent = myplot(x, y, s)
        #         ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        #         ax.set_title("Smoothing with  $\sigma$ = %d" % s)
        #
        #
        # plt.savefig('tmp_img/heat_map_'+dataset+'_'+architecture)
        if task == 'roc_using_si_and_top1':
            roc_data[dataset+architecture+'fpr_cutoff'] = fpr
            test_center_acc_bi = bin(test_center_acc, cutoff=test_center_acc_cutoff)
            fpr, tpr, _ = roc_curve(test_center_acc_bi, 1-test_si)
            roc_auc = auc(fpr, tpr)


            roc_data[dataset+architecture+'fpr'] = fpr
            roc_data[dataset+architecture+'tpr'] = tpr
            roc_data[dataset+architecture+'roc'] = roc_auc

    if task in ['pred_using_top_1', 'roc_using_si_and_top1']:
        path_test_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_x4.npz'
        d_test = np.load(path_test_embed_and_neighbor_acc)
        path_train_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_train_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_50_x4.npz'
        d_train = np.load(path_train_embed_and_neighbor_acc)

        train_example_num = 0
        test_example_num = 0
        if dataset in ['cifar10']:
            train_example_num = 50000
            test_example_num = 10000
        elif dataset in ['fmnist']:
            train_example_num = 60000
            test_example_num = 10000
        elif dataset in ['svhn']:
            train_example_num = 73257
            test_example_num = 26032

        test_center_acc = d_test['test_neighbor_avg_acc'][:test_example_num]
        train_center_acc = d_train['train_neighbor_avg_acc'][:train_example_num]

        # print(d_test['test_softmax'].shape, d_test['test_softmax'])
        test_top_1 = np.amax(d_test['test_softmax'], axis=1)
        train_top_1 = np.amax(d_train['train_softmax'], axis=1)


        train_center_acc_cutoff = 0.75
        test_center_acc_cutoff = train_center_acc_cutoff

        from scipy import stats
        train_perc = stats.percentileofscore(train_center_acc, train_center_acc_cutoff)/100

        # train_top_1_cutoff = np.quantile(train_top_1, train_perc)
        train_top_1_cutoff = train_center_acc_cutoff
        print('top_1_cutoff', train_top_1_cutoff)
        test_top_1_cutoff = train_top_1_cutoff

        # plt.hist(test_center_acc)
        # plt.show()


        top_conf_inds = set(np.argwhere(test_center_acc<=test_center_acc_cutoff).flatten())
        bottom_conf_inds = set(np.argwhere(test_center_acc>test_center_acc_cutoff).flatten())

        # print(test_center_acc.shape, np.min(test_center_acc), np.mean(test_center_acc))

        top_bias_inds = set(np.argwhere(test_top_1<=test_top_1_cutoff).flatten())
        bottom_bias_inds = set(np.argwhere(test_top_1>test_top_1_cutoff).flatten())

        precision, recall, fpr, m11, m01 = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds)

        top_bias_inds_r = set(np.random.choice(test_example_num, len(top_bias_inds), replace=False))
        bottom_bias_inds_r = set([i for i in range(test_example_num)]) - set(top_bias_inds_r)

        precision_r, recall_r, fpr_r, m11_r, m01_r = get_single_precision_recall_fpr_helper(top_conf_inds, bottom_conf_inds, top_bias_inds_r, bottom_bias_inds_r)
        f1 = est_f1(precision, recall)
        f1_r = est_f1(precision_r, recall_r)
        print('top1 & {:0.3f} & {:0.3f} & {:d} & {:d} \\\\'.format(f1, m11, m01))
        print('random & {:0.3f} & {:0.3f} & {:d} & {:d} \\\\'.format(f1_r, m11_r, m01_r))
        if task == 'roc_using_si_and_top1':
            test_center_acc_bi = bin(test_center_acc, cutoff=test_center_acc_cutoff)

            fpr, tpr, _ = roc_curve(test_center_acc_bi, 1-test_top_1)
            roc_auc = auc(fpr, tpr)


            roc_data[dataset+architecture+'fpr'+'_top1'] = fpr
            roc_data[dataset+architecture+'tpr'+'_top1'] = tpr

    if task == 'find_one_weak_diverse_region':
        path_test_embed_and_neighbor_acc = 'tmp_data_with_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_sweeping_rotate_x4_save_input.npz'
        d_test = np.load(path_test_embed_and_neighbor_acc)

        test_example_num = 0
        if dataset in ['cifar10']:
            test_example_num = 10000
        elif dataset in ['fmnist']:
            test_example_num = 10000
        elif dataset in ['svhn']:
            test_example_num = 26032

        # ['find_weak_region', 'find_strong_region', 'save_img']
        mode = 'find_weak_region'
        if mode == 'find_weak_region':
            test_center_acc = d_test['test_neighbor_avg_acc'][:test_example_num]
            test_si = get_si(d_test, test_example_num, 'test', 20)
            test_imgs = d_test['test_saved_transformed_images']
            test_y = d_test['test_y']
            test_y_pred = d_test['test_y_pred']

            inds = np.argwhere((test_si < 0.3) & (test_y[:test_example_num]==test_y_pred[:test_example_num]) & (test_y[:test_example_num]==2))
            print(inds.shape)
            ind = inds.squeeze()[0]
            ind_list = [(ind, 0, test_y[ind])]
            ys = set([test_y[ind]])
            for i in range(1, 21):
                cur_ind = ind + test_example_num*i
                cur_y_pred = test_y_pred[cur_ind]
                if cur_y_pred not in ys:
                    ys.add(cur_y_pred)
                    ind_list.append((cur_ind, i, cur_y_pred))
            print(ind_list)
            for cur_ind, i, cur_y_pred in ind_list:
                test_imgs[cur_ind][test_imgs[cur_ind]==np.array([0, 0, 0])] = 255
                plt.imshow(test_imgs[cur_ind])
                plt.show()

        elif mode == 'find_strong_region':
            test_center_acc = d_test['test_neighbor_avg_acc'][:test_example_num]
            test_y = d_test['test_y'][:test_example_num]
            test_si = get_si(d_test, test_example_num, 'test', 20)
            test_imgs = d_test['test_saved_transformed_images']

            inds = np.argwhere((test_si >= 1)&(test_y==2))

            ind = inds.squeeze()[1]
            ind_list = [(ind+i, 0, d_test['test_y'][ind]) for i in [0, 20000, 80000, 130000]]

            print(ind_list)
            for cur_ind, i, cur_y_pred in ind_list:
                test_imgs[cur_ind][test_imgs[cur_ind]==np.array([0, 0, 0])] = 255
                plt.imshow(test_imgs[cur_ind])
                plt.show()

        elif mode == 'save_img':
            #[(674, 0, 2), (20674, 2, 0), (80674, 8, 3), (130674, 13, 5)]
            #[(75, 0, 2), (20075, 0, 2), (80075, 0, 2), (130075, 0, 2)]
            test_img = d_test['test_saved_transformed_images']
            test_y = d_test['test_y']
            test_y_pred = d_test['test_y_pred']
            def save_fig(img, img_path):
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(img)
                fig.savefig(img_path, bbox_inches='tight', pad_inches=0)
            for ind, rot_str in zip([75+i for i in [0, 20000, 80000, 130000]], ['0', '2', '8', '13']):
                # test_img[ind][test_img[ind] == np.array([0., 0., 0.])] = 255
                save_fig(test_img[ind]/255, 'tmp_img/'+str(ind)+'_'+rot_str)
                # plt.imshow(d_test['test_saved_transformed_images'][ind]/255)
                # plt.show()


    if task == 'close_to_own_ratio':
        def find_nearest(array, value, j):
            array = np.asarray(array)
            idxs = (np.linalg.norm(array - value, axis=1)).argsort()
            idx = idxs[0]
            if idx == j:
                idx = idxs[1]
            return array[idx]

        num_of_classes = 10
        test_example_num = 0
        if dataset in ['cifar10']:
            test_example_num = 10000
        elif dataset in ['fmnist']:
            test_example_num = 10000
        elif dataset in ['svhn']:
            test_example_num = 26032
        path_test_embed_and_neighbor_acc = 'tmp_data_without_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_x4.npz'
        d = np.load(path_test_embed_and_neighbor_acc)

        d_neigh_acc = d['test_neighbor_avg_acc']
        d_y = d['test_y']
        d_embed = d['test_embed']
        d_y_pred = d['test_y_pred']

        th = 0.5

        d_embed_median = []
        for i in range(num_of_classes):
            d_embed_i = d_embed[d_y == i]
            d_embed_median.append(np.median(d_embed_i, axis=0))
        d_embed_median = np.asarray(d_embed_median)

        r_l = np.zeros(test_example_num)
        for j in range(test_example_num):
            d_c = find_nearest(d_embed_median, d_embed[j], d_y[j])
            r = np.linalg.norm(d_embed[j] - d_embed_median[d_y[j]]) / np.linalg.norm(d_embed[j] - d_c)
            r_l[j] = r


        strong_r_l = r_l[d_neigh_acc>=th]
        weak_r_l = r_l[d_neigh_acc<th]

        fig, axs = plt.subplots()
        axs.boxplot([strong_r_l, weak_r_l])
        plt.show()
        fig.savefig('tmp_img/'+dataset+'_'+architecture)

        #correct if the population S.D. is expected to be equal for the two groups.
        def cohen_d(x,y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

        print('{:0.3f}, {:0.3f}, {:0.3f}'.format(np.mean(weak_r_l), np.mean(strong_r_l), cohen_d(weak_r_l, strong_r_l)),  mannwhitneyu(strong_r_l, weak_r_l, alternative='two-sided'))


    if task == 'avg_ratio_w/s_d_to_centroid':
        path_test_embed_and_neighbor_acc = 'tmp_data_without_neighbor/'+dataset+'/natural_test_embed_and_neighbor_acc_'+architecture+'_random_rotate_and_shift_x4.npz'
        d = np.load(path_test_embed_and_neighbor_acc)

        num_of_classes = 10
        rs = []
        rs_j = []
        d_neigh_acc = d['test_neighbor_avg_acc']
        d_y = d['test_y']
        d_embed = d['test_embed']
        d_y_pred = d['test_y_pred']
        th = np.quantile(d_neigh_acc, 0.5)
        # print(d_neigh_acc[:100])
        # print('test_embed shape', d_embed.shape)
        # print('rob acc', (np.mean(d['test_neighbor_avg_acc'])*101 - np.mean(d_y == d_y_pred))/100)

        d_embed_median = []
        for i in range(num_of_classes):
            d_embed_i = d_embed[d_y == i]
            d_embed_median.append(np.median(d_embed_i, axis=0))

        for i in range(num_of_classes):

            d_neigh_acc_i = d_neigh_acc[d_y==i]
            d_embed_i = d_embed[d_y == i]

            d_embed_i_strong = d_embed_i[d_neigh_acc_i > th]
            d_embed_i_weak = d_embed_i[d_neigh_acc_i <= th]

            if len(d_embed_i_strong) > 0 and len(d_embed_i_weak) > 0:

                # ratio of avg dist to own class center
                r_i = np.mean(np.linalg.norm(d_embed_i_weak - d_embed_median[i], axis=1)) / np.mean(np.linalg.norm(d_embed_i_strong - d_embed_median[i], axis=1))
                rs.append(r_i)

                # ratio of avg dist to other class center
                r_j = 0
                for j in range(num_of_classes):
                    if j != i:
                        w_to_j_d = np.mean(np.linalg.norm(d_embed_i_weak - d_embed_median[j], axis=1))
                        s_to_j_d = np.mean(np.linalg.norm(d_embed_i_strong - d_embed_median[j], axis=1))

                        r_j += (w_to_j_d / s_to_j_d)
                        # print(i, j, w_to_j_d, s_to_j_d, r_j)
                r_j /= (num_of_classes - 1)
                rs_j.append(r_j)
        # print(rs)
        print(np.mean(rs))
        print(np.mean(rs_j))





    if task in ['prec_recall', 'prec_recall_changing_cutoff_y_pred', 'diff_training_mode']:
        path_test_embed_and_neighbor_acc = 'tmp_data_without_neighbor/'+dataset+'/natural_detector_test_result_nn_'+architecture+transformation_mode_str+'_x4'+retrain_with_weak_points_for_detector_str+'.npz'
        d = np.load(path_test_embed_and_neighbor_acc)
        d_y = d['test_labels_vec']
        d_y_pred = d['test_predictions_vec']
        d_y_pred = d_y_pred[:d_y.shape[0]]

        if task == 'prec_recall':
            cutoff_percs = [0.1, 0.5]
            for cutoff_perc in cutoff_percs:
                cutoff = np.quantile(d_y, cutoff_perc)

                precision, recall, _, m11, m01, precision_r, recall_r, _, m11_r, m01_r = get_single_precision_recall_fpr(d_y, d_y_pred, cutoff)

                print(cutoff, '{0:.3f}'.format(np.mean(np.abs(d_y-d_y_pred))))
                print('ours:', '{0:.3f}'.format(precision), '{0:.3f}'.format(recall), m11, m01)
                print('random:', '{0:.3f}'.format(precision_r), '{0:.3f}'.format(recall_r), m11_r, m01_r)
        elif task == 'prec_recall_changing_cutoff_y_pred':
            print(np.min(d_y), np.min(d_y_pred))
            for cutoff_y_pred in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                precision, recall, _, m11, m01, precision_r, recall_r, _, m11_r, m01_r = get_single_precision_recall_fpr(d_y, d_y_pred, 0.5, cutoff_y_pred)
                print(cutoff_y_pred, '{0:.3f}'.format(np.mean(np.abs(d_y-d_y_pred))), '{0:.3f}'.format(precision), '{0:.3f}'.format(recall), m11, m01)
        elif task == 'diff_training_mode':
            cutoff = np.quantile(d_y, 0.5)
            precision, recall, _, m11, m01, precision_r, recall_r, _, m11_r, m01_r = get_single_precision_recall_fpr(d_y, d_y_pred, cutoff)
            f1 = 2*(precision*recall)/(precision+recall)
            print(cutoff, '{0:.3f}'.format(np.mean(np.abs(d_y-d_y_pred))), '{0:.3f}'.format(f1))
            print('ours:', '{0:.3f}'.format(precision), '{0:.3f}'.format(recall), m11, m01)
            print('random:', '{0:.3f}'.format(precision_r), '{0:.3f}'.format(recall_r), m11_r, m01_r)

    elif task == 'diff_neighbor_num':
        d = np.load('tmp_data_without_neighbor/'+dataset+'/natural_train_embed_and_neighbor_acc_resnet_random_rotate_and_shift_'+str(num_of_transformation_for_train)+'_x4.npz')
        print(d['train_neighbor_avg_acc'][:20])


if task == 'roc_using_si_and_top1':
    y_num = len(datasets)
    x_num = len(architectures)
    fig, axs = plt.subplots(3, 3, sharex='col', sharey='row',
                gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,12))

    for i, dataset in enumerate(datasets):
        for j, architecture in enumerate(architectures):
            fpr = roc_data[dataset+architecture+'fpr']
            tpr = roc_data[dataset+architecture+'tpr']
            fpr_top1 = roc_data[dataset+architecture+'fpr'+'_top1']
            tpr_top1 = roc_data[dataset+architecture+'tpr'+'_top1']
            roc_auc = roc_data[dataset+architecture+'roc']

            axs[i][j].text(0.22, 0.15, dataset+','+architecture, fontsize=18)
            axs[i][j].text(0.22, 0.05, 'auc: {:.2f}'.format(roc_auc), fontsize=18)
            axs[i][j].plot(fpr, tpr, color='darkorange',
                     lw=2, label='ours')
            axs[i][j].plot(fpr_top1, tpr_top1, color='green',
                     lw=2, label='top1', linestyle=':')
            axs[i][j].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random')
            axs[i][j].set_xlim([0.0, 1.15])
            axs[i][j].set_ylim([0.0, 1.15])

            axs[i][j].tick_params(axis='both', which='major', labelsize=18)
            axs[i][j].tick_params(axis='both', which='minor', labelsize=18)

            axs[i][j].axvline(roc_data[dataset+architecture+'fpr_cutoff'], color='red', lw=2, linestyle='--')

            if i == 1:
                axs[i][j].set_ylabel('True Positive Rate', fontsize=18)
            if j == 1:
                axs[i][j].set_xlabel('False Positive Rate', fontsize=18)
            if i == y_num-1 and j == x_num-1:
                axs[i][j].legend(loc="center right", prop={'size': 18}, fancybox=True, framealpha=0.25)

    for ax in axs.flat:
        ax.label_outer()
    plt.savefig('tmp_img/div_roc_auc_50.pdf')


if task == 'spearmanr_si_neighbor_acc_wrt_neighbor_num':






    fig, ax = plt.subplots(1, 1, figsize=(12,12))

    colors = ['black', 'gray', 'rosybrown', 'red', 'darkcyan', 'gold', 'green', 'blue', 'purple']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    cnt = 0
    for dataset in datasets:
        for architecture in architectures:
            ys = np.array(spear[dataset+architecture]).squeeze()
            xs = np.array(list(range(2, 51, 3)))
            ax.plot(xs, ys, color=colors[cnt], label=dataset+'_'+architecture, lw=8, linestyle=linestyles[cnt])
            cnt += 1
    ax.set_xlabel('Number of neighbors used for estimating '+r'$\lambda$', fontsize=40)
    ax.set_ylabel('Spearman corrleation coeff', fontsize=40)
    ax.tick_params(labelsize=40)
    ax.legend(prop=dict(size=40))

    # code for legend outside
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(prop=dict(size=40), loc='center left', bbox_to_anchor=(1, 0.5))
    # ----------------------------
    ax.axvline(x=15, ymin=0, ymax=1, linestyle='--')
    if not os.path.exists('tmp_img'):
        os.mkdir('tmp_img')
    plt.savefig('tmp_img/spearmanr_si_neighb_acc.pdf', bbox_inches = 'tight',
    pad_inches = 0)
