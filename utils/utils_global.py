# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]unibas[dot]ch"
    __creation__ = "2025"
"""
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def vnorm(v):
    """Returns the L2 norm of the input vector v."""
    return np.sqrt(np.sum(np.square(v)))


def draw_frame(img, m=1, c=0):
    """Draw a frame of size m within the margins of img with color c."""
    mask_c = np.ones(img.shape[:2] + (3,), dtype=bool)
    in_slice = slice(m, -1 * m)
    mask_c[in_slice, in_slice, :] = 0
    img = (1.0 - mask_c) * img + mask_c * c
    return img


def text2np(text, size=12, color=(0, 0, 0)):
    # -- Empty image
    img_base = 255.0 * np.ones([size, int(size * len(text) / 1.5), 3])
    canvas_pil = PIL.Image.fromarray(img_base.astype(np.uint8))

    drawer = PIL.ImageDraw.Draw(canvas_pil)
    font = PIL.ImageFont.load_default()

    drawer.text((0, 0), text, fill=color, font=font, align="left")
    canvas_pil = np.asarray(canvas_pil)
    return canvas_pil


def concat_horizontal(list_images):
    list_h = [v.shape[0] for v in list_images]
    h_max = np.amax(list_h)
    sep_w = 255 * np.ones([h_max, 10, 3])

    list_images_ext = []
    for v in list_images:
        ext_c = 255 * np.ones([h_max - v.shape[0], v.shape[1], v.shape[2]])
        ext_c = np.concatenate([v, ext_c], axis=0)
        list_images_ext.append(ext_c)
        list_images_ext.append(sep_w)

    return np.concatenate(list_images_ext, axis=1)


def concat_vertical(list_images):
    list_w = [v.shape[1] for v in list_images]
    w_base = list_w[0]

    list_images_ext = []
    for v in list_images:
        if v.shape[1] > w_base:
            v = v[:, :w_base, :]
        ext_c = 255 * np.ones([v.shape[0], w_base - v.shape[1], v.shape[2]])
        ext_c = np.concatenate([v, ext_c], axis=1)
        list_images_ext.append(ext_c)

    return np.concatenate(list_images_ext, axis=0)
