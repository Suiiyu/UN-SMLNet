from utils import *
import tensorflow as tf
from keras import backend as K


def CE_Dice_loss():
    def computed_loss(y_true, y_pred):
        n_dims = y_pred.shape[-1]
        CE_loss = 0.
        Dice_loss = dice_coef_loss(y_true, y_pred)
        #############################################
        # gt = produce_mask_background( gt, y_pred)
        # gt = tf.stop_gradient( gt )
        #############################################
        for i in range(n_dims):
            gti = y_true[:, :, :, i]
            predi = y_pred[:, :, :, i]
            #Dice_loss += 1- dice_score(gti, predi)
            CE_loss = CE_loss - gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        loss = tf.reduce_mean(CE_loss) + Dice_loss
        return loss
    return computed_loss


def dice_coef_loss(y_true, y_pred):
    # input size：（batch_size, w, h, num_cls）
    # output size：（0）

    n_dims = y_pred.shape[-1]
    dice_loss = 0
    for i in range(n_dims):
        gti = y_true[:, :, :, i]
        predi = y_pred[:, :, :, i]
        intersection = K.sum(gti * predi, axis=(1,2))
        summation = K.sum(gti, axis=(1,2)) + K.sum(predi, axis=(1,2))
        # 整个batch求均值
        dice = K.mean((2. * intersection + 1e-5) / (summation + 1e-5), axis=0)
        dice_loss += 1-dice
    return dice_loss


# class-weighted cross-entropy loss and dice loss function
def softmax_weighted_loss():
    def computed_loss(y_true, y_pred):
        dice_loss = dice_coef_loss(y_true, y_pred)
        n_dims = y_pred.shape[-1]
        loss = 0.
        for i in range(n_dims):
            gti = y_true[:,:,:,i]
            predi = y_pred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(y_true))
            loss = loss -tf.reduce_mean(weighted*gti*tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return loss + dice_loss
    return computed_loss


def CE_Dice_loss_with_js(js):
    def compute_loss(y_true, y_pred):
        dice_loss = dice_coef_loss(y_true, y_pred)
        n_dims=y_pred.shape[-1]
        loss = 0.
        #############################################
        # gt = produce_mask_background( gt, y_pred)
        # gt = tf.stop_gradient( gt )
        #############################################
        for i in range(n_dims):
            gti = y_true[:,:,:,i]
            predi = y_pred[:,:,:,i]
            jsi = js[i]
            loss = loss -tf.reduce_mean(tf.math.exp(jsi)*gti * tf.log(tf.clip_by_value(predi, 0.005, 1)))+\
                tf.reduce_mean(jsi)

        return loss + dice_loss
    return compute_loss


def get_loss_sym_js(js):
    right_un_wce_loss = CE_Dice_loss_with_js(js)
    left_un_wce_loss = CE_Dice_loss_with_js(js)

    loss_metric = [right_un_wce_loss, left_un_wce_loss]

    loss_weight_metric = [1.0, 1.0]

    return loss_metric, loss_weight_metric
