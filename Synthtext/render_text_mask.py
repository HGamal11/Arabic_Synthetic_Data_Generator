

import os
import cv2
import math
import numpy as np
import pygame, pygame.locals


names_to_labels = {u'..':0,u'ا':1,u'أ':1,u'إ':1,u'آ':1,u'0':2,u'1':3,u'2':4,u'3':5,u'4':6,u'5':7,u'6':8,u'7':9,u'8':10,u'9':11,u'ب':12,u'پ':12,\
                   u'ت':13,u'ث':14,u'ج':15,u'چ':15,u'ح':16,u'خ':17,u'د':18,u'ذ':19,u'ر':20,u'ز':21,u'س':22,u'ش':23,u'ص':24,u'ض':25,u'ط':26,u'ظ':27,\
                   u'ع':28,u'غ':29,u'ف':30,u'ق':31,u'ك':32,u'ل':33,u'م':34,u'ن':35,u'ه':36,u'ة':36,u'ھ':36,u'و':37,u'ؤ':37,u'ي':38,u'ء':38,u'ئ':38,\
                   u'ى':38,u'ﺪ':39,u'لا':40,u'لآ':40}

def center2size(surf, size):

    canvas = np.zeros(size).astype(np.uint8)
    size_h, size_w = size
    surf_h, surf_w = surf.shape[:2]
    canvas[(size_h-surf_h)//2:(size_h-surf_h)//2+surf_h, (size_w-surf_w)//2:(size_w-surf_w)//2+surf_w] = surf
    return canvas


def crop_safe(arr, rect, bbs=[], pad=0):
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    v0 = [max(0,rect[0]), max(0,rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]), min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i,0] -= v0[0]
            bbs[i,1] -= v0[1]
        return arr, bbs
    else:
        return arr

def render_normal(font, text, textw):
    # get the number of lines
    lines = text.split('\n')
    lengths = [len(l) for l in lines]

    linesw = textw.split('\n')
    cls = []
    for c in linesw[0]:
        cl = names_to_labels[c]
        cls.append(cl)
    cls = np.array(cls)
    cls = np.flip(cls)

    # font parameters:
    line_spacing = font.get_sized_height() + 1

    # initialize the surface to proper size:
    line_bounds = font.get_rect(lines[np.argmax(lengths)])
    fsize = (round(2.0 * line_bounds.width), round(1.25 * line_spacing * len(lines)))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    surf1 = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    bbs = []
    bbs1 = []
    space = font.get_rect('O')
    space1 = font.get_rect('O')

    x, y = 0, 0
    x1, y1 = 0, 0

    for l in lines:
        x = 0 # carriage-return
        y += line_spacing # line-feed

        x1 = 0 # carriage-return
        y1 += line_spacing # line-feed

        for ch,cl in zip(l,cls): # render each character

            if ch.isspace(): # just shift
                x += space.width
                x1 += space1.width

            else:
                # render the character
                ch_bounds = font.render_to(surf, (x,y), ch)
                ch_bounds.x = x + ch_bounds.x
                ch_bounds.y = y - ch_bounds.y
                x += ch_bounds.width
                bbs.append(np.array(ch_bounds))

                ch_bounds1 = font.render_to(surf1, (x1,y1), ch, (cl,cl,cl))
                ch_bounds1.x = x1 + ch_bounds1.x
                ch_bounds1.y = y1 - ch_bounds1.y
                x1 += ch_bounds1.width
                bbs1.append(np.array(ch_bounds1))

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    r01 = pygame.Rect(bbs1[0])
    rect_union1 = r01.unionall(bbs1)

    # get the words:
    words = ' '.join(text.split())

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    bbs1 = np.array(bbs1)

    surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
    surf_arr = surf_arr.swapaxes(0,1)

    surf_arr1, bbs1 = crop_safe(pygame.surfarray.array3d(surf1), rect_union1, bbs1, pad=5)
    surf_arr1 = surf_arr1.swapaxes(0,1)

    #self.visualize_bb(surf_arr,bbs)
    return surf_arr, surf_arr1, bbs

def render_curved(font, text, textw, curve_rate, curve_center = None):
    wl = len(text)
    isword = len(text.split()) == 1

    cls = []
    for c in textw:
        cl = names_to_labels[c]
        cls.append(cl)
    cls = np.array(cls)
    cls = np.flip(cls)


    # create the surface:
    lspace = font.get_sized_height() + 1
    lbound = font.get_rect(text)
    #fsize = (round(2.0*lbound.width), round(3*lspace))
    fsize = (round(3.0*lbound.width), round(5*lspace))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    surf1 = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    # baseline state
    if curve_center is None:
        curve_center = wl // 2
    curve_center = max(curve_center, 0)
    curve_center = min(curve_center, wl - 1)
    mid_idx = curve_center #wl//2
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(wl)]
    curve[mid_idx] = -np.sum(curve) / max(wl-1, 1)
    rots  = [-int(math.degrees(math.asin(2 * curve_rate * (i-mid_idx)/(font.size/2)))) for i in range(wl)]

    bbs = []
    bbs1 = []
    # place middle char
    rect = font.get_rect(text[mid_idx])
    rect.centerx = surf.get_rect().centerx
    rect.centery = surf.get_rect().centery + rect.height
    rect.centery +=  curve[mid_idx]
    ch_bounds = font.render_to(surf, rect, text[mid_idx], rotation = rots[mid_idx])
    ch_bounds.x = rect.x + ch_bounds.x
    ch_bounds.y = rect.y - ch_bounds.y
    mid_ch_bb = np.array(ch_bounds)

    cl = cls[mid_idx]

    rect1 = font.get_rect(text[mid_idx])
    rect1.centerx = surf1.get_rect().centerx
    rect1.centery = surf1.get_rect().centery + rect1.height
    rect1.centery +=  curve[mid_idx]
    ch_bounds1 = font.render_to(surf1, rect1, text[mid_idx],(cl,cl,cl), rotation = rots[mid_idx])
    ch_bounds1.x = rect1.x + ch_bounds1.x
    ch_bounds1.y = rect1.y - ch_bounds1.y
    mid_ch_bb1 = np.array(ch_bounds1)

    # render chars to the left and right:
    last_rect = rect
    ch_idx = []

    last_rect1 = rect1
    ch_idx1 = []

    for i in range(wl):
        #skip the middle character
        if i == mid_idx:
            bbs.append(mid_ch_bb)
            ch_idx.append(i)

            bbs1.append(mid_ch_bb)
            ch_idx1.append(i)
            continue

        if i < mid_idx: #left-chars
            i = mid_idx-1-i
        elif i == mid_idx + 1: #right-chars begin
            last_rect = rect
            last_rect1 = rect1

        ch_idx.append(i)
        ch_idx1.append(i)

        ch = text[i]
        cl = cls[i]

        newrect = font.get_rect(ch)
        newrect.y = last_rect.y

        newrect1 = font.get_rect(ch)
        newrect1.y = last_rect1.y

        if i > mid_idx:
            newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
            newrect1.topleft = (last_rect1.topright[0] + 2, newrect1.topleft[1])

        else:
            newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
            newrect1.topright = (last_rect1.topleft[0] - 2, newrect1.topleft[1])

        newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
        newrect1.centery = max(newrect1.height, min(fsize[1] - newrect1.height, newrect1.centery + curve[i]))
        try:
            bbrect = font.render_to(surf, newrect, ch, rotation = rots[i])
            bbrect1 = font.render_to(surf1, newrect1, ch, (cl, cl, cl), rotation = rots[i])

        except ValueError:
            bbrect = font.render_to(surf, newrect, ch)
            bbrect1 = font.render_to(surf1, newrect, ch, (cl, cl, cl))

        bbrect1.x = newrect1.x + bbrect1.x
        bbrect1.y = newrect1.y - bbrect1.y
        bbs1.append(np.array(bbrect1))
        last_rect1 = newrect1

        bbrect.x = newrect.x + bbrect.x
        bbrect.y = newrect.y - bbrect.y
        bbs.append(np.array(bbrect))
        last_rect = newrect


    # correct the bounding-box order:
    bbs_sequence_order = [None for i in ch_idx]
    bbs_sequence_order1 = [None for i in ch_idx1]

    for idx,i in enumerate(ch_idx):
        bbs_sequence_order[i] = bbs[idx]
        bbs_sequence_order1[i] = bbs1[idx]

    bbs = bbs_sequence_order
    bbs1 = bbs_sequence_order1

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    r01 = pygame.Rect(bbs1[0])
    rect_union1 = r01.unionall(bbs1)
    # crop the surface to fit the text:
    bbs = np.array(bbs)
    bbs1 = np.array(bbs1)

    surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad = 5)
    surf_arr = surf_arr.swapaxes(0,1)

    surf_arr1, bbs = crop_safe(pygame.surfarray.array3d(surf1), rect_union1, bbs1, pad = 5)
    surf_arr1 = surf_arr1.swapaxes(0,1)

    return surf_arr, surf_arr1, bbs

def center_warpPerspective(img, H, center, size):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype = np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img = cv2.warpPerspective(img, M, size,
                    cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    return img

def center_pointsPerspective(points, H, center):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype = np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    return M.dot(points)

def perspective(img, rotate_angle, zoom, shear_angle, perspect, pad): # w first

    rotate_angle = rotate_angle * math.pi / 180.
    shear_x_angle = shear_angle[0] * math.pi / 180.
    shear_y_angle = shear_angle[1] * math.pi / 180.
    scale_w, scale_h = zoom
    perspect_x, perspect_y = perspect
    
    H_scale = np.array([[scale_w, 0, 0],
                        [0, scale_h, 0],
                        [0, 0, 1]], dtype = np.float32)
    H_rotate = np.array([[math.cos(rotate_angle), math.sin(rotate_angle), 0],
                         [-math.sin(rotate_angle), math.cos(rotate_angle), 0],
                         [0, 0, 1]], dtype = np.float32)
    H_shear = np.array([[1, math.tan(shear_x_angle), 0],
                        [math.tan(shear_y_angle), 1, 0], 
                        [0, 0, 1]], dtype = np.float32)
    H_perspect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [perspect_x, perspect_y, 1]], dtype = np.float32)

    H = H_rotate.dot(H_shear).dot(H_scale).dot(H_perspect)

    img_h, img_w = img.shape[:2]
    img_center = (img_w / 2, img_h / 2)
    points = np.ones((3, 4), dtype = np.float32)
    points[:2, 0] = np.array([0, 0], dtype = np.float32).T
    points[:2, 1] = np.array([img_w, 0], dtype = np.float32).T
    points[:2, 2] = np.array([img_w, img_h], dtype = np.float32).T
    points[:2, 3] = np.array([0, img_h], dtype = np.float32).T
    perspected_points = center_pointsPerspective(points, H, img_center)
    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]
    canvas_w = int(2 * max(img_center[0], img_center[0] - np.min(perspected_points[0, :]), 
                      np.max(perspected_points[0, :]) - img_center[0])) + 10
    canvas_h = int(2 * max(img_center[1], img_center[1] - np.min(perspected_points[1, :]), 
                      np.max(perspected_points[1, :]) - img_center[1])) + 10

    canvas = np.zeros((canvas_h, canvas_w), dtype = np.uint8)
    tly = (canvas_h - img_h) // 2
    tlx = (canvas_w - img_w) // 2
    canvas[tly:tly+img_h, tlx:tlx+img_w] = img
    canvas_center = (canvas_w // 2, canvas_h // 2)
    canvas_size = (canvas_w, canvas_h)
    canvas = center_warpPerspective(canvas, H, canvas_center, canvas_size)

    loc = np.where(canvas > 0)
    miny, minx = np.min(loc[0]), np.min(loc[1])
    maxy, maxx = np.max(loc[0]), np.max(loc[1])
    text_w = maxx - minx + 1
    text_h = maxy - miny + 1
    resimg = np.zeros((text_h + pad[2] + pad[3], text_w + pad[0] + pad[1])).astype(np.uint8)
    resimg[pad[2]:pad[2]+text_h, pad[0]:pad[0]+text_w] = canvas[miny:maxy+1, minx:maxx+1]
    return resimg

def render_text(font, text, textw, param):
    
    if param['is_curve']:
        return render_curved(font, text, textw, param['curve_rate'], param['curve_center'])
    else:
        return render_normal(font, text, textw)
