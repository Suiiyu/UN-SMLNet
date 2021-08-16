from keras.layers import Conv2D, BatchNormalization, Dropout, Concatenate, Activation,LeakyReLU
from keras.layers import GlobalAveragePooling2D, Reshape, Multiply, Add, UpSampling2D

# m: input
# ouput_dim: the num of channel
# res: controls the res connection
# drop: controls the dropout layer
# initpara: initial parameters
def convblock(m, ouput_dim, layername, res=False, drop=0, conv_num=3,**kwargs):
    n = Conv2D(filters=ouput_dim, name=layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001,name=layername + '_conv1_BN')(n)
    n = Activation(activation='relu', name=layername+'_conv1_ReLU')(n)

    n = Conv2D(filters=ouput_dim, name=layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001,name=layername + '_conv2_BN')(n)
    n = Activation(activation='relu', name=layername + '_conv2_ReLU')(n)
    if conv_num==3:
        n = Conv2D(filters=ouput_dim, name=layername + '_conv3', **kwargs)(n)
        n = BatchNormalization(momentum=0.95, epsilon=0.001, name=layername + '_conv3_BN')(n)
        n = Activation(activation='relu', name=layername + '_conv3_ReLU')(n)
    # n = Dropout(drop, trainable=True)(n) if drop else n
    n = Dropout(drop)(n) if drop else n
    return Concatenate()([m, n]) if res else n

# cwb模块
def se_layer(input, input_shape, ratio, layer_name):
    h, w, c = int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, c))(se)
    se = Conv2D(filters=c // ratio, kernel_size=1, strides=1, padding='valid',
                           name=layer_name + 'c_excitation1_conv')(se)
    #c_excitation1 = BatchNormalization(momentum=0.95, epsilon=0.001, name=layer_name + '_c_excitation1_BN')(
    #    c_excitation1)
    se = Activation(activation='relu', name=layer_name + '_c_excitation1_ReLU')(se)

    se = Conv2D(filters=c, kernel_size=1, strides=1, padding='valid',
                           name=layer_name + 'c_excitation2_conv')(se)
    #c_excitation2 = BatchNormalization(momentum=0.95, epsilon=0.001, name=layer_name + '_c_excitation2_BN')(
    #    c_excitation2)
    se = Activation(activation='sigmoid', name=layer_name + '_c_excitation2_Sigmoid')(se)

    se = Multiply()([input, se])
    se = Add()([input, se])
    # channel_sem = Lambda(channel_sem, K.sum(channel_sem, axis=3, keepdims=True))
    return se

def conv_att(m, input_shape, ratio, layer_name):
    c = int(input_shape[3])
    n = Conv2D(filters=c//ratio,kernel_size=3,strides=1,padding='same',
                       name=layer_name+'conv_att_conv1')(m)
    n = Activation(activation='relu', name=layer_name + 'conv_att_relu')(n)
    n = Conv2D(filters=c, kernel_size=1, strides=1, padding='same',
                            name=layer_name + 'conv_att_conv2')(n)
    n = Activation(activation='sigmoid', name=layer_name + 'conv_att_sigmoid')(n)
    n = Multiply()([m, n])
    n = Add()([m, n])
    return n


def scse_block(input, input_shape, ratio, layer_name):
    c = int(input_shape[3])
    # cSE
    c_squeeze = GlobalAveragePooling2D()(input)
    c_reshape = Reshape((1, 1, c))(c_squeeze)
    c_excitation1 = Conv2D(filters=c // ratio, kernel_size=1, strides=1, padding='valid',
                           name=layer_name + 'c_excitation1_conv')(c_reshape)
    # c_excitation1 = BatchNormalization(momentum=0.95, epsilon=0.001, name=layer_name + '_c_excitation1_BN')(
    #    c_excitation1)
    c_excitation1 = Activation(activation='relu', name=layer_name + '_c_excitation1_ReLU')(c_excitation1)

    c_excitation2 = Conv2D(filters=c, kernel_size=1, strides=1, padding='valid',
                           name=layer_name + 'c_excitation2_conv')(c_excitation1)
    # c_excitation2 = BatchNormalization(momentum=0.95, epsilon=0.001, name=layer_name + '_c_excitation2_BN')(
    #    c_excitation2)
    c_excitation2 = Activation(activation='sigmoid', name=layer_name + '_c_excitation2_Sigmoid')(c_excitation2)

    channel_sem = Multiply()([input, c_excitation2])
    #channel_sea = Add()([input, channel_sem])
    # sSE
    s_squeeze = Conv2D(filters=1, kernel_size=1, strides=1, padding='valid',
                           name=layer_name + 's_squeeze_conv')(input)
    s_squeeze_sigmoid = Activation(activation='sigmoid', name=layer_name + '_s_squeeze_sigmoid')(s_squeeze)

    spatial_sem = Multiply()([input, s_squeeze_sigmoid])
    output = Add()([channel_sem, spatial_sem])
    return output

def pam(input, ):
    #ref:DANet CVPR 2019
    pass
# 使用卷积进行pooling
def ConvMaxPooling(m,ouput_dim,layername):
    n= Conv2D(filters=ouput_dim, kernel_size=3, strides=2, padding='same', name=layername+'_conv')(m)
    #n = BatchNormalization(momentum=0.95, epsilon=0.001, name=layername+'_BN')(n)
    n = Activation(activation='relu', name=layername+'_ReLU')(n)
    return n


def convblock_leaky(m, ouput_dim, layername, res=False, drop=0, conv_num=3,**kwargs):
    n = Conv2D(filters=ouput_dim, name=layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001,name=layername + '_conv1_BN')(n)
    n = LeakyReLU(name=layername+'_conv1_LReLU')(n)

    n = Conv2D(filters=ouput_dim, name=layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001,name=layername + '_conv2_BN')(n)
    n = LeakyReLU(name=layername+'_conv2_LReLU')(n)

    if conv_num==3:
        n = Conv2D(filters=ouput_dim, name=layername + '_conv3', **kwargs)(n)
        n = BatchNormalization(momentum=0.95, epsilon=0.001, name=layername + '_conv3_BN')(n)
    #n = Activation(activation='relu', name=layername + '_conv3_ReLU')(n)
    # n = Dropout(drop, trainable=True)(n) if drop else n
    n = Dropout(drop)(n) if drop else n
    return Concatenate()([m, n]) if res else n


# 上采样后跟一个1*1卷积
def conv2DBN_Up(m, ouput_dim, up_shape, layername):
    if up_shape!=None:
        n = UpSampling2D(size=up_shape,interpolation='bilinear')(m)
    else: n = m
    n = Conv2D(filters=ouput_dim, kernel_size=1, padding='same', name=layername+'_conv')(n)
    #n = BatchNormalization(momentum=0.95, epsilon=0.001, name=layername+'_BN')(n)
    n = Activation(activation='relu', name=layername+'_ReLU')(n)
    return n


# 上采样后跟一个1*1卷积和前一层级输出相加
def conv2DBN_Up_addPre(m, m_pre, ouput_dim, up_shape, layername):
    if up_shape!=None:
        n = UpSampling2D(size=up_shape,interpolation='bilinear')(m)
    else: n = m

    n = Conv2D(filters=ouput_dim, kernel_size=1, padding='same', name=layername+'_conv')(n)
    n = Add(name=layername + 'add_pre_layer')([n, m_pre])
    #n = BatchNormalization(momentum=0.95, epsilon=0.001, name=layername+'_BN')(n)
    n = Activation(activation='relu', name=layername+'_ReLU')(n)
    return n


# BN层在激活层前。原先是将激活层内嵌在conv中，先于BN
def Conv2D_BN_AC(m, ouput_dim, kernel_size, layername):
    n = Conv2D(filters=ouput_dim, kernel_size=kernel_size, padding='same', name=layername)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001, name=layername+'_BN')(n)
    n = Activation(activation='relu', name=layername+'_ReLU')(n)
    return n


def boundary_refine_module(m, num_cls):

    n = Conv2D(filters=num_cls, kernel_size=3, padding='same')(m)
    n = Activation(activation='relu')(n)
    n = Conv2D(filters=num_cls, kernel_size=3, padding='same')(n)
    return Add()([m,n])


def global_conv_net_module(m, num_cls, k=3):

    l = Conv2D(filters=num_cls, kernel_size=(k,1), padding='same')(m)
    l = Conv2D(filters=num_cls, kernel_size=(1,k), padding='same')(l)

    r = Conv2D(filters=num_cls, kernel_size=(1,k), padding='same')(m)
    r = Conv2D(filters=num_cls, kernel_size=(k,1), padding='same')(r)

    return Add()([l,r])