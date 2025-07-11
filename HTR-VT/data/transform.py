import itertools
import cv2
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate, normalize
from torchvision.transforms import RandomCrop
import math

class Dilation:
    """
    OCR: stroke width increasing
    """
    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))


class Erosion:
    """
    OCR: stroke width decreasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))



class RandomTransform:
    """
    Random Transform adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, val):

        self.val = val

    def __call__(self, x):
        w, h = x.size

        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)

        def rd(d):
            return random.uniform(-d, d)

        def fd(d):
            return random.uniform(-dw, d)

        # generate a random projective transform
        # adapted from https://navoshta.com/traffic-signs-classification/
        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        tform = stf.ProjectiveTransform()
        tform.estimate(np.array((        #从对应点估计变换矩阵
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # determine shape of output image, to preserve size
        # trick take from the implementation of skimage.transform.rotate
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])

        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        # normalize
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        return Image.fromarray(x)


class SignFlipping:
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)


class DPIAdjusting:
    """
    Resolution modification
    """

    def __init__(self, factor, preserve_ratio):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)



class GaussianNoise:
    """
    Add Gaussian Noise
    """

    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        x_np = np.array(x)
        mean, std = np.mean(x_np), np.std(x_np)
        std = math.copysign(max(abs(std), 0.000001), std)
        min_, max_ = np.min(x_np,), np.max(x_np)
        normal_noise = np.random.randn(*x_np.shape)
        if len(x_np.shape) == 3 and x_np.shape[2] == 3 and np.all(x_np[:, :, 0] == x_np[:, :, 1]) and np.all(x_np[:, :, 0] == x_np[:, :, 2]):
            normal_noise[:, :, 1] = normal_noise[:, :, 2] = normal_noise[:, :, 0]
        x_np = ((x_np-mean)/std + normal_noise*self.std) * std + mean
        x_np = normalize(x_np, x_np, max_, min_, cv2.NORM_MINMAX)

        return Image.fromarray(x_np.astype(np.uint8))


class Sharpen:
    """
    Add Gaussian Noise
    """

    def __init__(self, alpha, strength):
        self.alpha = alpha
        self.strength = strength

    def __call__(self, x):
        x_np = np.array(x)
        id_matrix = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]]
                             )
        effect_matrix = np.array([[1, 1, 1],
                                  [1, -(8+self.strength), 1],
                                  [1, 1, 1]]
                                 )
        kernel = (1 - self.alpha) * id_matrix - self.alpha * effect_matrix
        kernel = np.expand_dims(kernel, axis=2)
        kernel = np.concatenate([kernel, kernel, kernel], axis=2)
        sharpened = cv2.filter2D(x_np, -1, kernel=kernel[:, :, 0])
        return Image.fromarray(sharpened.astype(np.uint8))


class ZoomRatio:
    """
        Crop by ratio
        Preserve dimensions if keep_dim = True (= zoom)
    """

    def __init__(self, ratio_h, ratio_w, keep_dim=True):
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.keep_dim = keep_dim

    def __call__(self, x):
        w, h = x.size
        x = RandomCrop((int(h * self.ratio_h), int(w * self.ratio_w)))(x)
        if self.keep_dim:
            x = x.resize((w, h), Image.BILINEAR)
        return x


class Tightening:
    """
    Reduce interline spacing
    """

    def __init__(self, color=255, remove_proba=0.75):
        self.color = color
        self.remove_proba = remove_proba

    def __call__(self, x):
        x_np = np.array(x)
        interline_indices = [np.all(line == 255) for line in x_np]
        indices_to_removed = np.logical_and(np.random.choice([True, False], size=len(x_np), replace=True, p=[self.remove_proba, 1-self.remove_proba]), interline_indices)
        new_x = x_np[np.logical_not(indices_to_removed)]
        return Image.fromarray(new_x.astype(np.uint8))