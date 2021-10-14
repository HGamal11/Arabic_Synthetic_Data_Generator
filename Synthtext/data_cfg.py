
import numpy as np

# font
font_size = [8, 50]
underline_rate = 0.01
strong_rate = .5
oblique_rate = 0.02
font_dir = 'Arfonts'

# text
text_filepath = 'arabic.txt'

# background
bg_filepath = 'labels.txt'

## background augment
brightness_rate = 0.8
brightness_min = 0.7
brightness_max = 1.3
color_rate = 0.8
color_min =0.7
color_max = 1.3
contrast_rate = 0.8
contrast_min = 0.7
contrast_max = 1.3

# curve
is_curve_rate = .2
curve_rate_param = [0.4, 0] # scale, shift for np.random.randn()

# perspective
rotate_param = [5, 0] # scale, shift for np.random.randn()
zoom_param = [0.1, 1] # scale, shift for np.random.randn()
shear_param = [5, 0] # scale, shift for np.random.randn()
perspect_param = [0.002, 0] # scale, shift for np.random.randn()

# render

## surf augment
elastic_rate = 0.001
elastic_grid_size = 4
elastic_magnitude = 2

## colorize
padding_ud = [0, 1]
padding_lr = [0, 1]
is_border_rate = 0.2
is_shadow_rate = 0.1
shadow_angle_degree = [1, 3, 5] # shift for shadow_angle_param
shadow_angle_param = [0.5, None] # scale, shift for np.random.randn()
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype = np.float32) # scale, shift for np.random.randn()
shadow_opacity_param = [0.1, 0.5] # shift for shadow_angle_param
color_filepath = 'data/colors_new.cp'
use_random_color_rate = 0.1
