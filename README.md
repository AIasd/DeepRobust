# DeepRobust
## Description:
This repo contains code for the paper **Understanding Local Robustness of Deep Neural Networks under Natural Variations** accepted to FASE 2021.

## Environment Setup:
```
conda env create -f scripts/environment.yml
```
## File Structures
The file structure looks like the following:
```
|--backup(need to be created manually)
    |--cifar10
    |--fmnist
    |--svhn
|--data(need to be created manually)
    |--cifar10
    |--fmnist
    |--svhn
    |--simulation (the online data path needs to be changed to this)
|--load(need to be created manually)
    |--cifar10
    |--fmnist
    |--svhn
    |--simulation (the online pretrained model path needs to be changed to this)
|--scripts
    |--analysis.py
    |--rq1.py
    |--train_classification_model.py
    |--train_detector.py
    |--tSNE.py
    |--utils_new.py

    |--dataloader
        |--loader.py
        |--preprocess_fmnist.py
    |--learning
        |--model_resnet.py
        |--model_wrn.py
        |--model_vgg16.py
        |--detector.py
    |--code-sdc
        ├── Automold.py
        ├── batch_generator.py
        ├── drive.py
        ├── Helpers.py
        ├── model_factory.py
        ├── models
        │   ├── abstract_model_provider.py
        │   ├── chauffeur.py
        │   ├── epoch.py
        │   ├── nvidia_dave2.py
        ├── train_self_driving_car.py
        ├── utils_train_self_driving_car.py
        ├── variational_autoencoder.py
        └── video.py
    |--configs: cmconfiguaration files
    |--tmp_data_with_neighbor (need to be created manually): consists of temporary files for doing analysis
    |--tmp_data_without_neighbor (need to be created manually): consists of temporary files for doing analysis
    |--tsne: consists of temporary files for doing analysis
```

## Datasets and Pretrained Models (only for steering angle prediction now; other models need to be trained by the users)
cifar10: https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-10 python version)

svhn: http://ufldl.stanford.edu/housenumbers/ (all three files in Format2)

fmnist: https://www.kaggle.com/zalando-research/fashionmnist#fashion-mnist_test.csv

steering angle prediction: https://academictorrents.com/details/221c3c71ac0b09b1bb31698534d50168dc394cc7

## Reproduction Instruction:
It should be noted that some small fluctuations of results might appear due to randomness.

### Preprocess data and train a classifier
1. Download each dataset online and put the data files into corresponding folder (e.g. one puts the bunch of files got from unzipping the cifar10 data zip file from its official website into the data/cifar10 folder).
2. Run
```
  cd scripts/dataloader
  python preprocess_fmnist.py
```
to process the fmnist dataset into a certain format if one wants to run experiment on it.
3. Train a classification model by running commands like the following (we use cifar10 and resnet here as an example):
```
  python train_classification_model.py --dataset=cifar10 --architecture=resnet
```
4. Move the trained model folder from `backup/cifar10` to `load/cifar10` and may need to remove the ending index if necessary e.g. if one gets the model folder with name `natural_resnet_2`, one needs to remove the `_2` in the end before moving it into load/cifar10.

### Extract feature vectors for neighbors and train/test a detector
5. Run
```
  python train_detector.py --dataset=cifar10 --architecture=resnet --running_mode=preprocess
```
to extract feature vectors for the trained resnet model on the neighbors of each of the original data point from both traing and testing set of the cifar10 dataset.
6. Run
```
  python train_detector.py --dataset=cifar10 --architecture=resnet --running_mode=train
```
to train the detector for the trained classifier on the cifar10 dataset.

7. (Table 4, Table 5) (need to change parameters accordingly and make sure the corresponding model and data have been run in the previous steps) Run
```
  python train_detector.py --dataset=cifar10 --architecture=resnet --running_mode=test
```
to test the detector for the trained classifier on the cifar10 dataset.

### Do extra analysis
Reproduce results for Figure 5 and Figure 8 (Need to change parameters inside. See the beginning comments in `tSNE.py`). Run:
```
python tSNE.py --dataset=cifar10 --architecture=resnet
```
Reproduce results for Table 2, Table 3, top1 baseline in Table 4 and Table 6, our method in Table 6 and Figure 7 (Need to change parameters accordingly inside the file and make sure the corresponding model and data have been run in the previous steps. See the beginning comments in `analysis.py`). Run:
```
python analysis.py
```
