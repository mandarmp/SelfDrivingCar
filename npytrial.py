import os
import cv2
import numpy as np
from keras import backend as K
from keras.utils import np_utils

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images

path1 = 'left'
path2 = 'right'
path3 = 'straight'
path4 = 'slow'
img_list1 = load_images_from_folder('C:\\Users\\ajeya\\PycharmProjects\\Self Driving Car\\data\\'+path1)
img_list2 = load_images_from_folder('C:\\Users\\ajeya\\PycharmProjects\\Self Driving Car\\data\\'+path2)
img_list3 = load_images_from_folder('C:\\Users\\ajeya\\PycharmProjects\\Self Driving Car\\data\\'+path3)
img_list4 = load_images_from_folder('C:\\Users\\ajeya\\PycharmProjects\\Self Driving Car\\data\\'+path4)
img_list = img_list1 + img_list2 + img_list3 + img_list4


# img_list.append(img_list)
# img_list.append(img_list2)
# img_list.append(img_list3)
# img_list.append(img_list4)
print(len(img_list1))
print(len(img_list2))
print(len(img_list3))
print(len(img_list4))
print(len(img_list))
num_channels = 1
num_classes = 3
img_data = np.array(img_list)
num_samples = img_data.shape[0]
labels = np.ones(num_samples, dtype='int64')
#print(type(labels))
labels[:3717] = 0
labels[6747:] = 2
labels[3717:6747] = 1

label_names = ['left','straight','right']
Y = np_utils.to_categorical(labels,num_classes)
print(Y)



print(img_data.shape)
#if num_channels == 1:
#    if K.image_dim_ordering() == 'th':
#        img_data = np.expand_dims(img_data, axis=1)
#        print(img_data.shape)  # (808,1,128,128)
#    else:
#        img_data = np.expand_dims(img_data, axis=4)
#        print(img_data.shape)  # (808,128,128,1)
#
#else:
#    if K.image_dim_ordering() == 'th':
#        img_data = np.rollaxis(img_data, 3, 1)
#        print(img_data.shape)


# data_dir_list = os.listdir('C:\\Users\\ajeya\\PycharmProjects\\Self Driving Car\\data')
#
# img_data_list = []
#
# for dataset in data_dir_list:
#     img_list = os.listdir('C:\\Users\\ajeya\\PycharmProjects\\Self Driving Car\\data')
#     print('images of {} loaded successfully'.format(dataset))
#
#     for img in img_list:
#         input_img = cv2.imread( dataset + '/' + img)
#         img_data_list.append(input_img)
#
#
# print(len(img_data_list))


# img_data = np.array(img_data_list)
# print(img_data.shape)