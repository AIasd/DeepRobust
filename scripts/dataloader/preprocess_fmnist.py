import os
import numpy as np

def preprocess_fmnist(folder_path, filename, mode):
    d = np.genfromtxt(os.path.join(folder_path, filename), delimiter=',')[1:]
    labels = d[:, 0]
    images = d[:, 1:].reshape((d.shape[0], 28, 28, 1))

    np.save(os.path.join(folder_path, mode+'_'+'images'), images)
    np.save(os.path.join(folder_path, mode+'_'+'labels'), labels)


if __name__ == '__main__':
    preprocess_fmnist('../../data/fmnist/', 'fashion-mnist_train.csv', 'train')
    preprocess_fmnist('../../data/fmnist/', 'fashion-mnist_test.csv', 'test')
