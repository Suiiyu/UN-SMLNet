from keras import backend as K
from keras.layers import Cropping2D
import numpy as np
from measures import hd,hd95,dc,jc,assd

from skimage import measure
from matplotlib import pyplot as plt
import os


def crop(tensors):
    '''
    :param tensors: List of two tensors, the second tensor having larger spatial dims
    :return:
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape( t )
        h_dims.append( h )
        w_dims.append( w )
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h // 2, crop_h // 2 + rem_h)
    crop_w_dims = (crop_w // 2, crop_w // 2 + rem_w)
    cropped = Cropping2D( cropping=(crop_h_dims, crop_w_dims) )( tensors[1] )
    return cropped


def center_crop(data, crop_size):
    x_in, y_in, z_in = data.shape
    x_out, y_out, z_out = crop_size[0], crop_size[1], crop_size[2]
    # if x_in<=x_out:
    # center crop
    x_offset = (x_in - x_out) // 2
    y_offset = (y_in - y_out) // 2
    z_offset = (z_in - z_out) // 2
    cropped_data = data[x_offset:(x_offset + x_out),
                   y_offset:(y_offset + y_out),
                   z_offset:(z_offset + z_out)]

    return cropped_data


def padding(data, out_shape):
    x_out, y_out, z_out = out_shape
    print(data.shape)
    print(out_shape)
    x_in, y_in, z_in = data.shape[0], data.shape[1], data.shape[2]
    x_pad = (x_out - x_in) // 2
    y_pad = (y_out - y_in) // 2
    z_pad = (z_out - z_in) // 2

    data_pad = np.pad(data, ((x_pad, x_pad), (y_pad, y_pad), (z_pad, z_pad)), 'constant')
    return data_pad


def dice_score(y_true, y_pred):
    eps = 1e-5
    axes = (1, 2)
    n_dims = y_pred.shape
    if n_dims==4:
        # 输入维度：（batch_size, w, h, num_cls）
        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    dice = K.mean((2. * intersection + eps) / (summation + eps), axis=0)
    return dice


def kullback_leibler_divergence_custom(y_true, y_pred):

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / (y_pred+K.epsilon())), axis=-1)


def JS_metric(right, left):
    # return shape:[bs,h,w]
    m = (right + left)/2.0
    kl_1 = kullback_leibler_divergence_custom(right,m)
    kl_2 = kullback_leibler_divergence_custom(left,m)
    JS = ((kl_1+kl_2)/2.0)
    return JS


def un_threshold_func(pred_map, un_map, un_ratio):
    # un_map: the uncertainty map
    # un_th: [0,1]
    min_value = np.amax(un_map)
    max_value = np.amax(un_map)
    un_th = un_ratio*(max_value-min_value)
    threshold_map = np.where(un_map>un_th,0,1)
    pred_map_thre = pred_map * threshold_map
    return pred_map_thre



def kl_custom(right,left):
    y_true = np.clip(right, 1e-5, 1)
    y_pred = np.clip(left, 1e-5, 1)
    # return np.sum(y_true * np.log(y_true / y_pred), axis=-1)
    return y_true * np.log(y_true /(y_pred+1e-5))

def JS_metric_test(right, left):
    m = (right + left) / 2.0
    kl_1 = kl_custom(right, m)
    kl_2 = kl_custom(left, m)
    JS = (kl_1 + kl_2) / 2.0
    return JS


def js_uncertainty(pred):
    pred_right = pred[0]
    pred_left = pred[1]
    js_uncertainty_volume = np.zeros((88,288,288))
    for slice_index in range(88):
        pred_fg_right_slice = pred_right[np.newaxis, slice_index, :, :]
        pred_fg_left_slice = pred_left[np.newaxis, slice_index, :, :]
        js = JS_metric_test(pred_fg_right_slice,pred_fg_left_slice)
        js_uncertainty_volume[slice_index,:,:] = js
    return js_uncertainty_volume


def get_metrics(y_true, y_pred, save_map=False, save_path=None):

    dice_medpy = dc(y_pred, y_true)
    print('dice_medpy: %f' % (dice_medpy))
    jaccard_medpy=jc(y_pred,y_true)
    print('jaccard_medpy: %f' % (jaccard_medpy))
    hd95_medpy = hd95(y_pred, y_true, voxelspacing=0.625, connectivity=1)
    print('hd95_medpy: %f' % ( hd95_medpy))
    assd_medpy = assd(y_pred, y_true, voxelspacing=0.625, connectivity=1)
    print('assd_medpy: %f' % (assd_medpy))
    return [dice_medpy, jaccard_medpy, hd95_medpy, assd_medpy]


def plot_seg2img(pred_volume,img_volume,save_path, dice_list=None, isvalue=False,cr='red'):
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    if cr=='red':
        color=[1.0,0.0,0.0]
    elif cr=='green':
        color=[0.0,1.0,0.0]
    elif cr=='blue':
        color=[0.0,0.0,1.0]
    else:
        assert 'unkown color'
    for i in range(88):
        pred_img = pred_volume[i,:,:]
        if len(img_volume.shape)!=4:
            img = img_volume[i,:,:,np.newaxis]#288,288,1
        else:
            img = img_volume[i,...]#288,288,1
        img = np.tile(img,(1,1,3))#288,288,3
        height, width = img.shape[:2]
        _, ax = plt.subplots(1, figsize=(height, width))
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')

        # masked_pred = np.ma.masked_where(pred_img==0,pred_img)
        alpha=0.5
        for ch in range(img.shape[-1]):
            img[:,:,ch] = np.where(pred_img==1,(img[:,:,ch]*(1-alpha)+alpha*color[ch])*255,img[:,:,ch]*255)

        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height/37.5, width/37.5)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if isvalue:
            plt.text(230,270,s=str(dice_list[i]),fontdict=dict(fontsize=20,color='w',family='Times New Roman',weight='bold'))

        ax.imshow(img.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, str(i)))
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def plot_seg2img_withcontour(pred_volume,img_volume,gt_volume, save_path, orientation='axial', dice_list=None, isvalue=False,pred_cr='blue', gt_cr='red'):
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)

    if pred_cr == 'red':
        pred_color = [1.0, 0.0, 0.0]
    elif pred_cr == 'green':
        pred_color = [0.0, 1.0, 0.0]
    elif pred_cr == 'blue':
        pred_color = [0.0, 0.0, 1.0]
    else:
        assert 'unkown color'

    if orientation=='axial':
        pred_volume = pred_volume  #(z,x,y)
        gt_volume = gt_volume
        img_volume = img_volume
    elif orientation == 'coronal':
        pred_volume = np.transpose(pred_volume, (2, 1, 0))#(y,x,z)
        gt_volume = np.transpose(gt_volume, (2, 1, 0))
        img_volume = np.transpose(img_volume, (2, 1, 0))
    for i in range(pred_volume.shape[0]):
        pred_img = pred_volume[i, :, :]
        if len(img_volume.shape) != 4:
            img = img_volume[i, :, :, np.newaxis]#288,288,1
        else:
            img = img_volume[i, ...] #288,288,1
        img = np.tile(img, (1, 1, 3)) #288,288,3
        height, width = img.shape[:2]
        contours = find_contours(gt_volume[i, :, :], level=0.5)
        _, ax = plt.subplots(1, figsize=(height, width))
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')

        # masked_pred = np.ma.masked_where(pred_img==0,pred_img)
        alpha=0.5
        for ch in range(img.shape[-1]):
            img[:,:,ch] = np.where(pred_img==1,(img[:,:,ch]*(1-alpha)+alpha*pred_color[ch])*255,img[:,:,ch]*255)
            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], color=gt_cr, linewidth=2)
        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height/37.5, width/37.5)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if isvalue:
            plt.text(230,270,s=str(dice_list[i]),fontdict=dict(fontsize=20,color='w',family='Times New Roman',weight='bold'))

        ax.imshow(img.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, str(i)))
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def plot_s2s_volume(s2s_volume,save_path):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(s2s_volume.shape[1])
    y = np.arange(s2s_volume.shape[2])
    z = np.arange(s2s_volume.shape[0])
    X,Y,Z=np.meshgrid(x,y,z)

    img = ax.sca
    for i in range(640):
        for j in range(640):
            for k in range(88):
                c=s2s_volume[k,i,j]
                if c==0:
                    continue
                else:
                    plt.scatter(i, j, k, c=c, cmap=plt.cm.hot)
    fig.colorbar(img)
    #plt.show()
    plt.savefig()


def save_uncertainty_map(un_map, save_path,save_type='uncertainty'):
    n_dims = un_map.shape
    if len(n_dims) == 4:  # (88,288,288,2)
        for cls_index in range(n_dims[-1]):
            save_path_cls = os.path.join(save_path, str(cls_index))
            un_map_cls = un_map[..., cls_index]  # (88,288,288)
            save_slice_uncertainty(un_map_cls, save_path_cls, save_type=save_type)
    else:  # (88,288,288)
        save_slice_uncertainty(un_map, save_path, save_type=save_type)


def save_slice_uncertainty(un_map, save_path, save_type='uncertainty'):
    if save_type == 'error':
        cmap_used = plt.cm.OrRd
    elif save_type == 'uncertainty':
        cmap_used = plt.cm.OrRd
    elif save_type == 'probability':
        cmap_used = plt.cm.hot
    else:
        assert 'Unkown type'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    for slice_index in range(un_map.shape[0]):
        plt.figure()
        plt.imshow(un_map[slice_index, :, :], cmap=cmap_used, interpolation='nearest')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        #   plt.axes('off')
        save_name = os.path.join(save_path, str(slice_index) + '.png')
        plt.savefig(save_name)
        plt.clf()
        plt.cla()
        plt.close()


def cca(input_volume):
    # 计算连通分量
    # 给不同的联通域赋予不同的值
    connect_label = measure.label(input_volume, connectivity=2, return_num=False)
    # 计算每个联通分量的面积，标注为0的联通分量会忽略
    connect_area = measure.regionprops(connect_label)
    connect_box = []
    for connect_index in range(len(connect_area)):
        #存储每个区域的面积
        connect_box.append(connect_area[connect_index].area)
    max_connect_index = connect_box.index(max(connect_box))+1 #提取最大面积的编号
    connect_label[connect_label != max_connect_index] = 0
    connect_label[connect_label == max_connect_index] = 1
    return connect_label.astype('uint8')


def error_map(pred, gt):
    #计算预测结果和ground truch之间的差异
    error_map = np.absolute((gt-pred))
    return error_map