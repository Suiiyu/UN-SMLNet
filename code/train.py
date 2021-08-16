'''

Total params: 12,927,092
Trainable params: 12,918,260
Non-trainable params: 8,832

'''

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.preprocessing.image import ImageDataGenerator
from utils import *
from model.UN_SMLNet import CrabNet
from dataprocess import *
from keras.utils.np_utils import to_categorical
from keras import backend as K
from surface_distance.surface_distance import metrics as surf_metric
import matplotlib.pyplot as plt
seed = 1234
np.random.seed(seed)

def lr_ep_decay(model, base_lr, curr_ep, step=0.1):
    lrate = base_lr * step ** (curr_ep / 20)
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)


DATA_PATH = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_volumes/')


def dice_coef_online(img1, img2):
    eps = 1e-5
    ob_intersection = np.sum(img1 * img2)
    ob_summation = np.sum(img1) + np.sum(img2)
    if ob_summation == 0:
        ob_dice = 1.0
    else:
        ob_dice = (2.0 * ob_intersection + eps) / (ob_summation + eps)

    return ob_dice

if __name__ == '__main__':

    lr = 0.001
    crop_shape = (88, 288, 288)  # z,y,x
    resize_shape = None  # (88,256,256)
    input_shape = (288, 288, 1)
    num_cls = 2
    epochs = 10
    mini_batch_size = 4
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
    weights = None  # model_save_path +('/model_epoch_100.h5')
    model = CrabNet(input_shape, num_cls, lr, maxpool=True, isres=False, droprate=0,
                    weights=weights, isUN=isUN, isDE=isDE, cwb=cwb, ratio=ratio)

    volume_lists = get_volume_files(TRAIN_DATA_PATH, shuffle=False)#[:2]
    # radom select val volumes from the volume list
    val_ratio = 0.1
    train_volume_lists, val_volume_lists = split_volume_lists(volume_lists, val_ratio)
    img_train, mask_train = load_data(train_volume_lists, TRAIN_DATA_PATH, crop_size=crop_shape,
                                      resize_shape=resize_shape, nor_type='minmax')
    mask_train = to_categorical(mask_train, num_classes=num_cls)
    img_val_list = []
    mask_val_list = []
    for idx, val_name in enumerate(val_volume_lists):
        img_val_one, mask_val_one = load_data([val_volume_lists[idx]], TRAIN_DATA_PATH, crop_size=crop_shape,
                                  resize_shape=resize_shape,nor_type='minmax')  # 88,256,256,1
        mask_val_one = to_categorical(mask_val_one, num_classes=num_cls)
        img_val_list.append(img_val_one)
        mask_val_list.append(mask_val_one)


    print('train data shape:{}'.format(img_train.shape))  # ,img_val.shape))

    index_shuffle = np.arange(len(img_train))
    np.random.shuffle(index_shuffle)
    img_train = img_train[index_shuffle, :, :, :]
    mask_train = mask_train[index_shuffle, :, :, :]
    # data augementation
    kwargs = dict(
        rotation_range=20,
        zoom_range=[0.7, 1.3],
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                         batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                       batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)

    max_iter = (len(img_train) // mini_batch_size) * epochs
    step_iter = 1000
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    curr_ep = 0
    lrate = lr_ep_decay(model, base_lr, curr_ep, step=0.5)

    train_res = open(os.path.join(log_save_path, 'Train.txt'),  'a+')
    dev_res = open(os.path.join(log_save_path,'Val.txt'), 'a+')
    train_res.write('loss weight [1,1]\n %s\n' % (model.metrics_names))
    dev_res.write('%s \n' % (model.metrics_names))

    for e in range(epochs):
        print('Main Epoch {:d}'.format(e + 1))
        print('Learning rate: {:6f}'.format(lrate))
        train_result = []
        train_dice_list = []
        curr_ep = e + 1

        if (e + 1) % 20 == 0:
            lrate = lr_ep_decay(model, base_lr, curr_ep, step=0.5)

        for iteration in range(len(img_train) // mini_batch_size):
            img, mask = next(train_generator)

            if isDE:
                res = model.train_on_batch(img, mask)
            else:
                res = model.train_on_batch(img, [mask, mask])
            # res:loss,right_loss,left_loss,right_acc,right_dice,left_acc,left_dice
            curr_iter += 1
            train_result.append(res)

        train_result = np.array(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)

        train_res.write('%s\n' % (train_result))

        if (e + 1) % 2 == 0:
            # validation by volume
            dice_val = 0
            val_vol_index = 0
            for idx in range(len(val_volume_lists)):
                img_val = img_val_list[idx]
                mask_val = mask_val_list[idx]
                val_name = val_volume_lists[idx]
                pred_val = model.predict(img_val, batch_size=16)

                right_pred_val = np.argmax(pred_val[0], axis=-1)  # 88,256,256
                dice_coef_vol = dice_coef_online(np.argmax(mask_val, axis=-1), right_pred_val)
                print('val_name:{}, Dice:{}'.format(val_name, dice_coef_vol))
                dice_val += dice_coef_vol
                if idx == 0:
                    if isDE:
                        eval_val = model.evaluate(img_val,mask_val, batch_size=16)
                    else:
                        eval_val = model.evaluate(img_val, [mask_val, mask_val], batch_size=16)
                else:
                    if isDE:
                        eval_val = np.array(eval_val) + np.array(model.evaluate(img_val, mask_val, batch_size=16))
                    else:
                        eval_val = np.array(eval_val) + np.array(model.evaluate(img_val, [mask_val, mask_val], batch_size=16))
            dice_val /= len(val_volume_lists)
            eval_val = [l / len(val_volume_lists) for l in eval_val]
            dev_res.write('%s %s\n' % (eval_val, dice_val))
            print('Dev set mean predict dice: {}'.format(dice_val))

            save_file = '_'.join(
                ['model_epoch', str(e + 1)]) + '.h5'
            save_path = os.path.join(model_save_path, save_file)
            print('Saving model weights to {}'.format(save_path))
            model_json = model.to_json()
            with open(os.path.join(model_save_path, 'model_epoch' +str(e+1) + '.json'), 'w') as file:
                file.write(model_json)
            model.save_weights(save_path)

    train_res.close()
    dev_res.close()

