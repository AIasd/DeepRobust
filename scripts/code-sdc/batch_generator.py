from keras.utils import Sequence
# modfification
import numpy as np
from utils_train_self_driving_car import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, load_image, augment, preprocess, rgb2yuv


class Generator(Sequence):

    def __init__(self, path_to_pictures, steering_angles, is_training, args, mode='plain', dtheta=0):
        self.batch_size = args.batch_size
        self.path_to_pictures = path_to_pictures
        self.steering_angles = steering_angles
        self.is_training = is_training
        self.args = args
        self.mode = mode
        self.dtheta = dtheta

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        batch_paths = self.path_to_pictures[start_index:end_index]
        steering_angles = self.steering_angles[start_index:end_index]

        images = np.empty([len(batch_paths), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        # images = []
        steers = np.empty([len(batch_paths)])
        for i, paths in enumerate(batch_paths):
            center, left, right = batch_paths[i]
            steering_angle = steering_angles[i]
            # augmentation
            if self.is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(self.args.data_dir, center, left, right, steering_angle)
            else:
                image = load_image(self.args.data_dir, center)

            # add the image and steering angle to the batch
            images[i] = preprocess(image, self.mode, self.dtheta)
            # images[i] = image
            # images.append(image)
            steers[i] = steering_angle

        # images = preprocess(images, self.mode, self.dtheta)

        # from matplotlib import pyplot as plt
        # import cv2
        # img = images[-1]
        # img = cv2.cvtColor(images[-1].astype('uint8'), cv2.COLOR_YUV2RGB)
        # plt.imshow(img.astype('uint8'))
        # plt.savefig(self.mode)

        return images, steers

    def __len__(self):
        return len(self.path_to_pictures) // self.batch_size
