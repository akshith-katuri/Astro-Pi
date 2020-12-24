# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:48:45 2020

@author: Akshith
"""

import matplotlib.pyplot as plt
from skimage import io, color
from PIL import Image, ImageDraw
from skimage import img_as_float
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.feature import canny
import numpy as np

# earth_pic_pillow = Image.open("zz_astropi_1_photo_170.jpg")

# im_a = Image.new("L", earth_pic_pillow.size, 0)
# draw = ImageDraw.Draw(im_a)
# draw.ellipse((139, 826, 2420, 965), fill=255)

# earth_pic_pillow_copy = earth_pic_pillow.copy()
# earth_pic_pillow_copy.putalpha(im_a)
# earth_pic_pillow_crop = earth_pic_pillow_copy.crop((139, 826, 2420, 965))
# earth_pic_pillow_crop.save('zz_astropi_1_photo_170.png')
def percent_cloud(file):
    earth_pic = io.imread(file)
    earth_pic_grey = color.rgb2gray(earth_pic)
    
    edges = canny(earth_pic_grey, sigma=0)
    
    entropy_img = entropy(img_as_float(edges), disk(15))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    plt.imshow(binary)

    print("The percentage of clouds is:", (np.sum(binary==0)*100)/(np.sum(binary==0) + np.sum(binary==1)))

percent_cloud("zz_astropi_1_photo_200.jpg")