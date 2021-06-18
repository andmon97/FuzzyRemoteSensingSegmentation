import cv2
import os
import csv
from skimage.filters.rank import entropy
from skimage.morphology import disk

imagesPath = 'WHDLD/Images'
labeledImagesPath = 'WHDLD/ImagesPNG'

width = 256
height = 256

countBuilding = 0
countRoad = 0
countVegetation = 0
countWater = 0

imagesExt = '.jpg'
labelsExt = '.png'
i=0
# Opening file
fileImgs = open('WHDLD/reduced4ClassV3/labels.txt', 'r')

# Writing to file
with open('pixelClassificationDataset/reducedV3.csv', mode='w',newline='') as x_file:
    x_writer = csv.writer(x_file, delimiter=';')

    for line in fileImgs:
        imgName = line.strip()
        # read image
        img = cv2.imread(os.path.join(imagesPath, imgName + imagesExt), cv2.IMREAD_COLOR)
        #dim = (new_width, new_heigth)
        # resize image
        #resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # read segmented for labels
        imgLabs = (cv2.imread(os.path.join(labeledImagesPath, imgName + labelsExt), cv2.IMREAD_COLOR))

        grayScaleImg = cv2.imread(os.path.join(imagesPath, imgName + imagesExt), cv2.IMREAD_GRAYSCALE)
        blurImg = cv2.GaussianBlur(grayScaleImg, (9, 9), 0)
        imgEntropy = entropy(blurImg, disk(5))

        for y in range(height):
            for x in range(width):
                # print(resized_img[y, x], end="\t")
                # pivel values from images
                pixel = img[y, x]
                blue = str(pixel[0])
                green = str(pixel[1])
                red = str(pixel[2])

                #pixel values from segmented "parallel" images
                pixelLab = imgLabs[y, x]
                blueLab = (pixelLab[0])
                greenLab = (pixelLab[1])
                redLab = (pixelLab[2])
                if redLab == 255 and greenLab == 0 and blueLab == 0:
                    label = 0 # building
                    countBuilding += 1
                elif (redLab == 255 and greenLab == 255 and blueLab == 0) \
                        or (redLab == 192 and greenLab == 192 and blueLab == 0):
                    label = 1  # road
                    countRoad += 1
                elif redLab == 0 and greenLab == 255 and blueLab == 0:
                    label = 2  # vegetation
                    countVegetation += 1
                else:
                    label = 3  # water
                    countWater += 1

                # pixel values from entropy "parallel" images
                entropyValPix = str(imgEntropy[y, x])

                x_writer.writerow([red,green,blue,entropyValPix,label])

        print(i)
        i += 1

# Closing files
fileImgs.close()

print("% Pixel per class: ")
print("Building ", countBuilding/(countBuilding+countRoad+countVegetation+countWater))
print("Road ", countRoad/(countBuilding+countRoad+countVegetation+countWater))
print("Vegetation ", countVegetation/(countBuilding+countRoad+countVegetation+countWater))
print("Water ", countWater/(countBuilding+countRoad+countVegetation+countWater))


