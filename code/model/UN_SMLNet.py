from keras import optimizers
from keras.models import Model
from keras.layers import MaxPooling2D,Input
from custom_losses import *
from utils import *
from custom_blocks import *
'''

Total params: 11,207,152
Trainable params: 11,198,320
Non-trainable params: 8,832

'''

def CrabNet(input_shape, num_cls, lr, maxpool=True, isres=False, droprate=0, weights=None,
            isUN=True, isDE=False, cwb='se', ratio=1):
    '''initialization'''
    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )
    num_cls = num_cls
    data = Input(shape=input_shape, dtype='float', name='data')
    # encoder
    enconv1 = convblock(data, ouput_dim=32, layername='block1', res=isres, drop=droprate, **kwargs)
    pool = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1')(enconv1) if maxpool \
        else ConvMaxPooling(enconv1, ouput_dim=32, layername='pool1')

    enconv2 = convblock(pool, ouput_dim=64, layername='block2', res=isres, drop=droprate, **kwargs)
    pool = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool2')(enconv2) if maxpool \
        else ConvMaxPooling(enconv2, ouput_dim=64, layername='pool2')

    enconv3 = convblock(pool, ouput_dim=128, layername='block3', res=isres, drop=droprate, **kwargs)
    pool = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool3')(enconv3) if maxpool \
        else ConvMaxPooling(enconv3, ouput_dim=128, layername='pool3')

    enconv4 = convblock(pool, ouput_dim=256, layername='block4', res=isres, drop=droprate, **kwargs)
    pool = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool4')(enconv4) if maxpool \
        else ConvMaxPooling(enconv4, ouput_dim=256, layername='pool4')

    enconv5 = convblock(pool, ouput_dim=512, layername='block5notl', res=isres, drop=droprate, **kwargs)

    # decoder
    up = conv2DBN_Up(enconv5, ouput_dim=256, up_shape=[2,2], layername='upsampling1')
    merge1 = Concatenate()([up, enconv4])
    deconv6 = convblock(merge1, ouput_dim=256, layername='deconv6', res=isres, drop=droprate, **kwargs)

    up = conv2DBN_Up(deconv6, ouput_dim=128, up_shape=[2,2], layername='upsampling2')
    merge2 = Concatenate()([up, enconv3])
    deconv7 = convblock(merge2, ouput_dim=128, layername='deconv7', res=isres, drop=droprate, **kwargs)

    up = conv2DBN_Up(deconv7, ouput_dim=64, up_shape=[2,2], layername='upsampling3')
    merge3 = Concatenate()([up, enconv2])
    deconv8 = convblock(merge3, ouput_dim=64, layername='deconv8', res=isres, drop=droprate, **kwargs)

    up = conv2DBN_Up(deconv8, ouput_dim=32, up_shape=[2,2], layername='upsampling4')
    merge4 = Concatenate()([up, enconv1])
    deconv9 = convblock(merge4, ouput_dim=32, layername='deconv9', res=isres, drop=0.5, **kwargs)

    # decoder
    if cwb == 'scse':
        deconv9_weight = scse_block(deconv9, deconv9.shape, ratio=16, layer_name='deconv9')
        leg8 = conv2DBN_Up(deconv9_weight, ouput_dim=num_cls, up_shape=None, layername='leg8')
        deconv8_weight = scse_block(deconv8, deconv8.shape, ratio=16, layer_name='deconv8')
        leg7 = conv2DBN_Up(deconv8_weight, ouput_dim=num_cls, up_shape=[2, 2], layername='leg7')
        deconv7_weight = scse_block(deconv7, deconv7.shape, ratio=16, layer_name='deconv7')
        leg6 = conv2DBN_Up(deconv7_weight, ouput_dim=num_cls, up_shape=[4, 4], layername='leg6')
        deconv6_weight = scse_block(deconv6, deconv6.shape, ratio=16, layer_name='deconv6')
        leg5 = conv2DBN_Up(deconv6_weight, ouput_dim=num_cls, up_shape=[8, 8], layername='leg5')
    elif cwb == 'se':
        deconv9_weight = se_layer(deconv9, deconv9.shape, ratio=16, layer_name='deconv9')
        leg8 = conv2DBN_Up(deconv9_weight, ouput_dim=num_cls, up_shape=None, layername='leg8')
        deconv8_weight = se_layer(deconv8, deconv8.shape, ratio=16, layer_name='deconv8')
        leg7 = conv2DBN_Up(deconv8_weight, ouput_dim=num_cls, up_shape=[2, 2], layername='leg7')
        deconv7_weight = se_layer(deconv7, deconv7.shape, ratio=16, layer_name='deconv7')
        leg6 = conv2DBN_Up(deconv7_weight, ouput_dim=num_cls, up_shape=[4, 4], layername='leg6')
        deconv6_weight = se_layer(deconv6, deconv6.shape, ratio=16, layer_name='deconv6')
        leg5 = conv2DBN_Up(deconv6_weight, ouput_dim=num_cls, up_shape=[8, 8], layername='leg5')
    elif cwb == 'conv_att':
        deconv9_weight = conv_att(deconv9, deconv9.shape, ratio=ratio, layer_name='deconv9')
        leg8 = conv2DBN_Up(deconv9_weight, ouput_dim=num_cls, up_shape=None, layername='leg8')
        deconv8_weight = conv_att(deconv8, deconv8.shape,  ratio=ratio, layer_name='deconv8')
        leg7 = conv2DBN_Up(deconv8_weight, ouput_dim=num_cls, up_shape=[2, 2], layername='leg7')
        deconv7_weight = conv_att(deconv7, deconv7.shape,  ratio=ratio, layer_name='deconv7')
        leg6 = conv2DBN_Up(deconv7_weight, ouput_dim=num_cls, up_shape=[4, 4], layername='leg6')
        deconv6_weight = conv_att(deconv6, deconv6.shape,  ratio=ratio, layer_name='deconv6')
        leg5 = conv2DBN_Up(deconv6_weight, ouput_dim=num_cls, up_shape=[8, 8], layername='leg5')
    elif cwb == None:
        leg8 = conv2DBN_Up(deconv9, ouput_dim=num_cls, up_shape=None, layername='leg8')
        leg7 = conv2DBN_Up(deconv8, ouput_dim=num_cls, up_shape=[2, 2], layername='leg7')
        leg6 = conv2DBN_Up(deconv7, ouput_dim=num_cls, up_shape=[4, 4], layername='leg6')
        leg5 = conv2DBN_Up(deconv6, ouput_dim=num_cls, up_shape=[8, 8], layername='leg5')


    merge_right = Concatenate(name='concate_right')([leg8, leg7, leg6, leg5])
    # encoder
    if cwb == 'scse':
        enconv4_weight = scse_block(enconv4, enconv4.shape, ratio=16, layer_name='enconv4')
        leg4 = conv2DBN_Up(enconv4_weight, ouput_dim=num_cls, up_shape=[8, 8], layername='leg4')
        enconv3_weight = scse_block(enconv3, enconv3.shape, ratio=16, layer_name='enconv3')
        leg3 = conv2DBN_Up(enconv3_weight, ouput_dim=num_cls, up_shape=[4, 4], layername='leg3')
        enconv2_weight = scse_block(enconv2, enconv2.shape, ratio=16, layer_name='enconv2')
        leg2 = conv2DBN_Up(enconv2_weight, ouput_dim=num_cls, up_shape=[2, 2], layername='leg2')
        enconv1_weight = scse_block(enconv1, enconv1.shape, ratio=16, layer_name='enconv1')
        leg1 = conv2DBN_Up(enconv1_weight, ouput_dim=num_cls, up_shape=None, layername='leg1')
    elif cwb == 'se':
        enconv4_weight = se_layer(enconv4, enconv4.shape, ratio=16, layer_name='enconv4')
        leg4 = conv2DBN_Up(enconv4_weight, ouput_dim=num_cls, up_shape=[8, 8], layername='leg4')
        enconv3_weight = se_layer(enconv3, enconv3.shape, ratio=16, layer_name='enconv3')
        leg3 = conv2DBN_Up(enconv3_weight, ouput_dim=num_cls, up_shape=[4, 4], layername='leg3')
        enconv2_weight = se_layer(enconv2, enconv2.shape, ratio=16, layer_name='enconv2')
        leg2 = conv2DBN_Up(enconv2_weight, ouput_dim=num_cls, up_shape=[2, 2], layername='leg2')
        enconv1_weight = se_layer(enconv1, enconv1.shape, ratio=16, layer_name='enconv1')
        leg1 = conv2DBN_Up(enconv1_weight, ouput_dim=num_cls, up_shape=None, layername='leg1')
    elif cwb == 'conv_att':
        enconv4_weight = conv_att(enconv4, enconv4.shape, ratio=ratio, layer_name='enconv4')
        leg4 = conv2DBN_Up(enconv4_weight, ouput_dim=num_cls, up_shape=[8, 8], layername='leg4')
        enconv3_weight = conv_att(enconv3, enconv3.shape, ratio=ratio, layer_name='enconv3')
        leg3 = conv2DBN_Up(enconv3_weight, ouput_dim=num_cls, up_shape=[4, 4], layername='leg3')
        enconv2_weight = conv_att(enconv2, enconv2.shape, ratio=ratio, layer_name='enconv2')
        leg2 = conv2DBN_Up(enconv2_weight, ouput_dim=num_cls, up_shape=[2, 2], layername='leg2')
        enconv1_weight = conv_att(enconv1, enconv1.shape, ratio=ratio, layer_name='enconv1')
        leg1 = conv2DBN_Up(enconv1_weight, ouput_dim=num_cls, up_shape=None, layername='leg1')
    elif cwb == None:
        leg4 = conv2DBN_Up(enconv4, ouput_dim=num_cls, up_shape=[8, 8], layername='leg4')
        leg3 = conv2DBN_Up(enconv3, ouput_dim=num_cls, up_shape=[4, 4], layername='leg3')
        leg2 = conv2DBN_Up(enconv2, ouput_dim=num_cls, up_shape=[2, 2], layername='leg2')
        leg1 = conv2DBN_Up(enconv1, ouput_dim=num_cls, up_shape=None, layername='leg1')

    merge_left = Concatenate(name='concate_left')([leg1, leg2, leg3, leg4])
    prediction_left_logit = Conv2D(filters=num_cls, kernel_size=1,
                             padding='same', name='predictions_left_logit')(merge_left)
    prediction_left = Activation(activation='softmax',name='prediction_left')(prediction_left_logit)

    prediction_right_logit = Conv2D(filters=num_cls, kernel_size=1,
                              padding='same', name='predictions_right_logit')(merge_right)
    prediction_right = Activation(activation='softmax', name='predictions_right')(prediction_right_logit)
    if isDE:
        # 只有一侧预测分支
        isUN = False
        model = Model(inputs=data, outputs=prediction_right)
    else:
        model = Model(inputs=data, outputs=[prediction_right, prediction_left])
    # model.load_weights('/Share1/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    if weights is not None:
        model.load_weights(weights, by_name=False)
    sgd = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if isUN:
        js = JS_metric(prediction_right,prediction_left)
        loss_metric, loss_weight_metric = get_loss_sym_js(js)
        # 只有右侧带有不确定性校正
        #loss_metric, loss_weight_metric = get_loss_kl(js)
        # 返回值有7项，总损失，右侧损失，左侧损失，右侧acc，右侧dice，左侧acc，左侧dice
        model.compile(optimizer=sgd, loss=loss_metric, loss_weights=loss_weight_metric,
                      metrics=['accuracy', dice_score])
    else:
        # 损失使用未加权的交叉熵和dice损失
        loss_metric = CE_Dice_loss()
        # 只有一个预测结果，返回值有3项，总损失，acc，dice
        model.compile(optimizer=sgd, loss=loss_metric, metrics=['accuracy', dice_score])
    model.summary()
    return model


if __name__ == '__main__':
    model= CrabNet((256, 256, 1), 1, 0.001, maxpool=True, weights=None)

