import numpy as np
import SimpleITK as sitk
import os, cv2
import random
from utils import center_crop
#random.expovariate()


def save_nrrd(save_path_fullname, ctrs, save_data, ref_data):
    if os.path.exists(save_path_fullname):
        pass
    else:
        os.makedirs(save_path_fullname)
    data = sitk.GetImageFromArray(save_data)
    data.SetDirection(ref_data.GetDirection())
    data.SetOrigin(ref_data.GetOrigin())
    data.SetSpacing(ref_data.GetSpacing())
    sitk.WriteImage(data, os.path.join(save_path_fullname, ctrs+'_pred.nrrd'))


def load_nrrd(full_path_filename, normalize='None'):
    data = sitk.ReadImage(full_path_filename)
    data = sitk.Cast(data, sitk.sitkFloat32)  # x,y,z
    if normalize == 'minmax':
        data_array = sitk.GetArrayFromImage(data)  # change data type: Image to array;z,y,x
        data_array = max_min_nor(data_array)
    elif normalize == 'zscore':
        data_array = sitk.GetArrayFromImage(data)  # change data type: Image to array;z,y,x
        data_array = zscore_norm(data_array)
    elif normalize == 'sitknorm':
        # normalize the data into (0,1)
        data = sitk.RescaleIntensity(data, outputMinimum=0, outputMaximum=1)
        data_array = sitk.GetArrayFromImage(data)  # change data type: Image to array;z,y,x
    else:
        data_array = sitk.GetArrayFromImage(data)  # change data type: Image to array;z,y,x

    return data_array, data


def max_min_nor(img):
    _max = np.amax(img)
    _min = np.amin(img)
    _range = _max-_min
    return (img-_min)/_range


def zscore_norm(img):
    # x = (x-mean)/std
    img = img * 1.0
    img_mean = np.mean(img, axis=(0, 1), keepdims=True)
    img_std = np.std(img, axis=(0, 1), keepdims=True)
    img -= img_mean
    img /= img_std + 0.00000000001
    return img


def load_data(volume_list, volume_data_path, crop_size=None, crop='center_crop', resize_shape=None,nor_type='minmax'):

    if resize_shape == None:
        data_list = np.zeros((len(volume_list)*crop_size[0], crop_size[1], crop_size[2], 1))
        label_list = np.zeros((len(volume_list)*crop_size[0], crop_size[1], crop_size[2], 1))
    else:
        data_list = np.zeros((len(volume_list)*resize_shape[0], resize_shape[1], resize_shape[2], 1))
        label_list = np.zeros((len(volume_list)*resize_shape[0], resize_shape[1], resize_shape[2], 1))
    for idx, volume in enumerate(volume_list):
        data, _ = load_nrrd(os.path.join(volume_data_path, volume, 'lgemri.nrrd'))
        label, _ = load_nrrd(os.path.join(volume_data_path, volume, 'laendo.nrrd'))
        if crop_size:
        # center crop according to crop size

            data = center_crop(data, crop_size)
            label = center_crop(label, crop_size)
        for slices in range(data.shape[0]):

            if nor_type == 'minmax':
                img = max_min_nor(data[slices, :, :])
            elif nor_type == 'zscore':
                img = zscore_norm(data[slices, :, :])
            elif nor_type == 'None':
                img = data[slices, :, :]
            else:
                assert 'Unkown normalization type!'
            lb = label[slices, :, :]
            if resize_shape != None and resize_shape != crop_size:
                img = cv2.resize(img, (resize_shape[1], resize_shape[2]), interpolation=cv2.INTER_NEAREST)
                data_list[idx*data.shape[0]+slices, :, :, 0] = img
                lb = cv2.resize(lb, (resize_shape[1], resize_shape[2]), interpolation=cv2.INTER_NEAREST)
                label_list[idx*data.shape[0]+slices, :, :, 0] = np.where(lb > 0.5, 1, 0).astype('uint8')
            else:
                data_list[idx * data.shape[0] + slices, :, :, 0] = img
                label_list[idx*data.shape[0]+slices, :, :, 0] = np.where(lb > 0.5, 1, 0).astype('uint8')
    return data_list, label_list


def save_image(save_path_fullname,ctrs,save_data):
    if os.path.exists(save_path_fullname):
        pass
    else:
        os.makedirs(save_path_fullname)
    for slices in range(save_data.shape[0]):
        img = save_data[slices,:,:]
        cv2.imwrite(os.path.join(save_path_fullname,'%s.png' % (ctrs+'_'+str(slices))), img)


def get_volume_files(volume_path, shuffle=True):
    # get the volume name list
    volume_list = os.listdir(volume_path)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(volume_list)
    print('Number of examples: {:d}'.format(len(volume_list)))
    return volume_list


def split_volume_lists(volume_list, val_ratio):
    # split the volume into train and val
    # :volume_list
    # :val_ratio, the ratio of val data
    # return train volumes and val volumes
    val_num = int(len(volume_list) * val_ratio)
    val_volume_list = random.sample(volume_list,val_num)
    for i in val_volume_list:
        volume_list.remove(i)
    train_volume_list = volume_list
    return train_volume_list, val_volume_list
