

import os
import cv2
import numpy as np
from pygame import freetype
import random
import Augmentor

from . import render_text_mask
from . import colorize
from . import data_cfg

from bidi.algorithm import get_display
import arabic_reshaper
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def make_ar_text(x,font):
    reshaper = arabic_reshaper.ArabicReshaper(arabic_reshaper.config_for_true_type_font(font,arabic_reshaper.ENABLE_ALL_LIGATURES))
    reshaped_text = reshaper.reshape(x)
    ar_text = get_display(reshaped_text)
    return ar_text

class datagen():

    def __init__(self):
        
        freetype.init()
        cur_file_path = os.path.dirname(__file__)
        
        font_dir = os.path.join(cur_file_path, data_cfg.font_dir)
        self.font_list = os.listdir(font_dir)
        self.font_list = [os.path.join(font_dir, font_name) for font_name in self.font_list]
        self.standard_font_path = os.path.join(cur_file_path, data_cfg.standard_font_path)
        
        color_filepath = os.path.join(cur_file_path, data_cfg.color_filepath)
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)
        
        text_filepath = os.path.join(cur_file_path, data_cfg.text_filepath)
        self.text_list = open(text_filepath, 'r',encoding="utf8").readlines()
        self.text_list = [text.strip() for text in self.text_list]
        
        bg_filepath = os.path.join(cur_file_path, data_cfg.bg_filepath)
        self.bg_list = open(bg_filepath, 'r').readlines()
        self.bg_list = [img_path.strip() for img_path in self.bg_list]
        
        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(probability = data_cfg.elastic_rate,
            grid_width = data_cfg.elastic_grid_size, grid_height = data_cfg.elastic_grid_size,
            magnitude = data_cfg.elastic_magnitude)
        
        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(probability = data_cfg.brightness_rate, 
            min_factor = data_cfg.brightness_min, max_factor = data_cfg.brightness_max)
        self.bg_augmentor.random_color(probability = data_cfg.color_rate, 
            min_factor = data_cfg.color_min, max_factor = data_cfg.color_max)
        self.bg_augmentor.random_contrast(probability = data_cfg.contrast_rate, 
            min_factor = data_cfg.contrast_min, max_factor = data_cfg.contrast_max)

    def gen_ar_data(self):
        while True:
            # choose font, text and bg
            font = np.random.choice(self.font_list)
            textw = np.random.choice(self.text_list)
            text = make_ar_text(textw,font)

            s = random.choice(self.bg_list)
            bg = cv2.imread(s)

            # init font
            font = freetype.Font(font)
            font.antialiased = True
            font.origin = True

            # choose font style
            font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
            font.underline = np.random.rand() < data_cfg.underline_rate
            font.strong = np.random.rand() < data_cfg.strong_rate
            font.oblique = np.random.rand() < data_cfg.oblique_rate

            # render text to surf
            param = {
                        'is_curve': np.random.rand() < data_cfg.is_curve_rate,
                        'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn() 
                                      + data_cfg.curve_rate_param[1],
                        'curve_center': np.random.randint(0, len(text))
                    }

            param['curve_center'] = int(param['curve_center'] / len(text))
            surf2, surf12, bbs2 = render_text_mask.render_text(font, text, textw, param)

            surf12 = surf12[:,:,0]  #*(255/38)

            seq = iaa.Sequential([
                #iaa.Sharpen((0.0, 1.0)),  # sharpen the image
                #iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Affine(
                    #scale={"x": (0.9, 1.2), "y": (0.9, 1.2)},
                    rotate=(-15, 15),
                    shear=(-2, 2),fit_output=True,mode='edge'),
                #iaa.LinearContrast((0.85, 1.15)),
                iaa.PerspectiveTransform(scale=(0.01, 0.1), fit_output=True,mode=cv2.BORDER_REPLICATE)], random_order=True)


            surf12 = SegmentationMapsOnImage(surf12, shape=surf2.shape)
            surf2, surf12 = seq(image=surf2, segmentation_maps=surf12)
            surf12 = surf12.get_arr()

            # choose a background
            surf2_h, surf2_w = surf2.shape[:2]

            surf2 = render_text_mask.center2size(surf2, (surf2_h, surf2_w))
            surf2_h, surf2_w = surf12.shape[:2]
            mask = render_text_mask.center2size(surf12, (surf2_h, surf2_w))
            mask = mask.clip(0, 40)

            bg_h, bg_w = bg.shape[:2]
            if bg_w < surf2_w or bg_h < surf2_h:
                continue
            x = np.random.randint(0, bg_w - surf2_w + 1)
            y = np.random.randint(0, bg_h - surf2_h + 1)
            t_b = bg[y:y+surf2_h, x:x+surf2_w, :]
            bgs = [[t_b]]
            self.bg_augmentor.augmentor_images = bgs
            t_b = self.bg_augmentor.sample(1)[0][0]
            min_h2 = np.min(bbs2[:, 3])

            # get font color
            if np.random.rand() < data_cfg.use_random_color_rate:
                fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)
            else:
                fg_col, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, t_b)

            # colorful the surf and conbine foreground and background
            param = {
                        'is_border': np.random.rand() < data_cfg.is_border_rate,
                        'bordar_color': tuple(np.random.randint(0, 256, 3)),
                        'is_shadow': np.random.rand() < data_cfg.is_shadow_rate,
                        'shadow_angle': np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree)
                                        + data_cfg.shadow_angle_param[0] * np.random.randn(),
                        'shadow_shift': data_cfg.shadow_shift_param[0, :] * np.random.randn(3)
                                        + data_cfg.shadow_shift_param[1, :],
                        'shadow_opacity': data_cfg.shadow_opacity_param[0] * np.random.randn()
                                          + data_cfg.shadow_opacity_param[1]
                    }
            _, img = colorize.colorize(surf2, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h2, param)

            break

        return [img, mask, textw]

