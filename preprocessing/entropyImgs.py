import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

imagesPath = 'WHDLD/Images'
imagesExt = '.jpg'
labelsExt = '.png'

# Opening file
fileImgs = open('WHDLD/reduced4ClassV3/labels.txt', 'r')

for line in fileImgs:
    imgName = line.strip()
    grayScaleImg = cv2.imread(os.path.join(imagesPath, imgName + imagesExt), cv2.IMREAD_GRAYSCALE)
    blurImg = cv2.GaussianBlur(grayScaleImg, (9, 9), 0)

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                                   sharex=True, sharey=True)

    img0 = ax0.imshow(blurImg, cmap=plt.cm.gray)
    ax0.set_title("Image")
    ax0.axis("off")
    fig.colorbar(img0, ax=ax0)

    img1 = ax1.imshow(entropy(blurImg, disk(5)), cmap='gray')
    ax1.set_title("Entropy")
    ax1.axis("off")
    fig.colorbar(img1, ax=ax1)

    fig.tight_layout()

    plt.show()