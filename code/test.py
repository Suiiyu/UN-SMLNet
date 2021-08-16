'''

Total params: 12,927,092
Trainable params: 12,918,260
Non-trainable params: 8,832

'''

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import *
from model.UN_SMLNet import CrabNet
from dataprocess import *

from surface_distance.surface_distance import metrics as surf_metric

seed = 1234
np.random.seed(seed)

DATA_PATH = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_volumes/')
TEST_SAVE_PATH = os.path.join('data/test_save')

TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_volumes/')


if __name__ == '__main__':

    lr = 0.001
    crop_shape = (88, 288, 288)  # z,y,x
    resize_shape = None  # (88,256,256)
    input_shape = (288, 288, 1)
    num_cls = 2
    isUN = True
    isDE = False
    ratio = 1
    cwb = 'conv_att'#None#'scse'#'se'#
    # model save path
    save_subpath = 'UN_SMLNet'
    log_save_path = os.path.join('logs', save_subpath)
    model_save_path = os.path.join('model_logs', save_subpath)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    weights = model_save_path +('/model_epoch_100.h5')
    model = CrabNet(input_shape, num_cls, lr, maxpool=True, isres=False, droprate=0,
                    weights=weights, isUN=isUN, isDE=isDE, cwb=cwb, ratio=ratio)

    test_res = open(os.path.join(log_save_path,'Test.txt'),'a+')
    test_res.write('loss weight:[1,1];dropout 0.5 in block 9; cca, block 3conv; test data.\n')
    test_res.write('%s\n' % (weights))

    test_volume_lists = get_volume_files(TEST_DATA_PATH, shuffle=False)[:1]
    for idx, test_name in enumerate(test_volume_lists):
        print('idx:', idx)
        print('test_name:', test_name)
        img_test, mask_test = load_data([test_volume_lists[idx]], TEST_DATA_PATH, crop_size=crop_shape,
                                        resize_shape=resize_shape, nor_type='minmax')  # 88,256,256,1

        pred_test = model.predict(img_test, batch_size=16)
        #uncertainty_map = js_uncertainty(pred_test.squeeze(axis=-1))

        #save_uncertainty_map(uncertainty_map, os.path.join(TEST_SAVE_PATH, 'uncertainty',save_subpath, test_name),
        #                     save_type='uncertainty')
        if isDE:
            right_pred_test = np.argmax(pred_test, axis=-1)
        else:
            right_pred_test = np.argmax(pred_test[0], axis=-1)# 仅一侧输出
            seg2img_save_path = os.path.join(TEST_SAVE_PATH, 'seg2img', 'SMLNet-Convatt', test_name)
            plot_seg2img(cca(right_pred_test.astype('uint8')), img_test, seg2img_save_path, cr='blue')


        mask_raw, mask_image = load_nrrd(os.path.join(TEST_DATA_PATH, test_name, 'laendo.nrrd'), normalize='minmax')
        mask_shape = mask_raw.shape
        volume_pred_test_crop_size = np.zeros((crop_shape))
        volume_pred_test_raw_size = np.zeros((mask_shape))
        if resize_shape!=None:
            for img_index in range(mask_shape[0]):
                volume_pred_test_crop_size[img_index, :, :] = cv2.resize(right_pred_test[img_index, :, :],
                                                                     (crop_shape[1], crop_shape[2]),
                                                                     interpolation=cv2.INTER_NEAREST)
            volume_pred_test_raw_size = padding(volume_pred_test_crop_size, mask_shape)
            del volume_pred_test_crop_size
        else:
            del volume_pred_test_crop_size
            volume_pred_test_raw_size = padding(right_pred_test, mask_shape)
        volume_pred_test_raw_size = cca(volume_pred_test_raw_size.astype('uint8'))

        metrics_raw = get_metrics(mask_raw, volume_pred_test_raw_size)
        volume_pred_test_raw_size_crop = center_crop(volume_pred_test_raw_size, crop_shape)
        mask_raw_crop = center_crop(mask_raw, crop_shape)

        surface_dis = surf_metric.compute_surface_distances(mask_raw.astype(np.bool),
                                                            volume_pred_test_raw_size.astype(np.bool),
                                                            spacing_mm=(0.625, 0.625, 0.625))

        surface_dice = surf_metric.compute_surface_dice_at_tolerance(surface_dis, 1)

        print(surface_dice)
        test_res.write('%s %s surface metrics:%s\n' % (test_name, metrics_raw, surface_dice))

        # save volume
        #save_nrrd(os.path.join(TEST_SAVE_PATH, 'volume'), test_name, (volume_pred_test_raw_size*255).astype('uint8'),mask_image)
        # save_nrrd(os.path.join(TEST_SAVE_PATH, 'mask_volume'), test_name, mask_raw,mask_image)
        #dice_test = dice_coef_online(mask_raw, volume_pred_test_raw_size)

    test_res.close()

