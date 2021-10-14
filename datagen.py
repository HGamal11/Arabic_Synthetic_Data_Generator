

import os
import cv2
import cfg
from Synthtext.gen import datagen
import numpy as np

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():

    makedirs(cfg.i_dir)
    makedirs(cfg.m_dir)
    f = open(cfg.txt_file,'w',encoding='UTF-8')
    mp_gen = datagen()

    i = 0

    for idx in range(0,cfg.sample_num):
        print ("Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))
        img, mask, text2 = mp_gen.gen_ar_data()

        i_path = os.path.join(cfg.i_dir, str(i) + '.png')
        m_path = os.path.join(cfg.m_dir, str(i) + '.png')

        cv2.imwrite(i_path, img,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(m_path, mask,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        line = str(i)+'.png,'+text2+'\n'
        f.writelines(line)

        i+=1

if __name__ == '__main__':
    main()
