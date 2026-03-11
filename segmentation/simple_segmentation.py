import os
import gc 
import numpy as np
from scipy import io
import json
from skimage.io import imsave, imread
from numpy.lib.stride_tricks import as_strided as ast
from glob import glob
from skimage.transform import resize
from skimage.filters import threshold_otsu
import scipy

import tensorflow as tf  # numerical operations on gpu
import tensorflow.keras.backend as K

def standardize(img):
    # standardization using adjusted standard deviation

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img) == 2:
        img = np.dstack((img, img, img))

    return img

def est_label_multiclass(image,M,MODEL,NCLASSES=4,TARGET_SIZE=(512,512)):

    est_label = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], NCLASSES))
    
    for counter, model in enumerate(M):
        try:
            if MODEL=='segformer':
                est_label = model(tf.expand_dims(image, 0)).logits
            else:
                est_label = tf.squeeze(model(tf.expand_dims(image, 0)))
        except:
            if MODEL=='segformer':
                est_label = model(tf.expand_dims(image[:,:,:3], 0)).logits
            else:
                est_label = tf.squeeze(model(tf.expand_dims(image[:,:,:3], 0)))
                
            # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
            est_label = est_label + est_label2 + est_label3 + est_label4

        K.clear_session()

    return est_label, counter
    
def do_seg_array(rgb_array, 
           model_list, 
           MODEL='segformer',
           NCLASSES=4, 
           N_DATA_BANDS=3, 
           TARGET_SIZE=(512,512), 
           gpu=0
           ):
    """segments an rgb array with segformer model

    inputs:
    rgb_array (np.ndarray): rgb array
    M 
    Model
    NCLASSES (int): 4, number of prediction classes
    N_DATA_BANDS (int): 3, R, G, B
    TARGET_SIZE ((int,int)): size of image
    """

    smallimage = resize(
        rgb_array, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    image = np.array(smallimage)
    w,h,bands = np.shape(image)
    image = tf.cast(smallimage, tf.uint8)

    try: ##>3 bands
        if image.shape[-1]>3:
            image = image[:,:,:3]
    except:
        pass

    image = standardize(image.numpy()).squeeze()

    if np.ndim(image)==2:
        image = np.dstack((image, image, image))
    image = tf.transpose(image, (2, 0, 1))


    if np.std(image)==0:

        print("Image {} is empty")
        est_label = np.zeros((w,h))

    else:
            
        est_label, counter = est_label_multiclass(image,model_list,MODEL)

        est_label /= counter + 1

        if not isinstance(est_label, np.ndarray):
            # If not, convert it to a numpy array
            est_label = est_label.numpy()
        # Now, convert to 'float32'
        est_label = est_label.astype('float32')

        if MODEL=='segformer':
            est_label = resize(est_label, (1, NCLASSES, TARGET_SIZE[0],TARGET_SIZE[1]), preserve_range=True, clip=True).squeeze()
            est_label = np.transpose(est_label, (1,2,0))
            est_label = resize(est_label, (w, h))
        else:
            est_label = resize(est_label, (w, h))
    
    softmax_scores = est_label.copy() 

    if np.std(image)>0:
        est_label = np.argmax(softmax_scores, -1)
    else:
        est_label = est_label.astype('uint8')
    
    est_label = scipy.ndimage.median_filter(est_label, size=10)
    class_label_colormap = [
            "#3366CC",
            "#DC3912",
            "#FF9900",
            "#109618",
            "#990099",
            "#0099C6",
            "#DD4477",
            "#66AA00",
            "#B82E2E",
            "#316395",
            "#ffe4e1",
            "#ff7373",
            "#666666",
            "#c0c0c0",
            "#66cdaa",
            "#afeeee",
            "#0e2f44",
            "#420420",
            "#794044",
            "#3399ff",
        ]
    try:
        color_label = label_to_colors(
            est_label,
            smallimage.numpy()[:, :, 0] == 0,
            alpha=128,
            colormap=class_label_colormap,
            color_class_offset=0,
            do_alpha=False,
        )
    except:
        try:
            color_label = label_to_colors(
                est_label,
                smallimage[:, :, 0] == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )
        except:
            color_label = label_to_colors(
                est_label,
                smallimage == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )    
    return est_label, color_label

def fromhex(n):
    """hexadecimal to integer"""
    return int(n, base=16)

def label_to_colors(
    img,
    mask,
    alpha, 
    colormap,  
    color_class_offset,  
    do_alpha):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """

    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask == 1] = (0, 0, 0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg

def binary_lab_to_color_lab(mask, image, num_classes=4):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    threshold_level = 1
    black_mask = mask < threshold_level
    mask[black_mask] = 5
    if num_classes==4:
        #no data
        color_mask[mask==5] = (0, 0, 0)
        # water
        color_mask[mask == 0, :] = (51, 102, 204)
        # whitewater
        color_mask[mask == 1, :] = (220, 57, 18)
        # sand
        color_mask[mask == 2, :] = (255, 153, 0)
        # other
        color_mask[mask == 3, :] = (16, 150, 24)
        # ???
        color_mask[mask == 4, :] = (16, 150, 24)
    return color_mask
 