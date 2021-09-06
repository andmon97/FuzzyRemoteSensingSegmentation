import cv2
import os
import csv
import pandas as pd
import regex as re

imagesPath = 'WHDLD/Images'
labeledImagesPath = 'WHDLD/ImagesPNG'

width = 256
height = 256

countBuilding = 0
countRoad = 0
countVegetation = 0
countWater = 0
countBareSoil = 0
countPavement = 0

imagesExt = '.jpg'
labelsExt = '.png'

df = pd.read_csv("WHDLD/multilabels.csv", sep=';')
road = df.loc[df['baresoil'] == 1]
roadImages = road[["IMAGE\LABEL"]].to_numpy()

# Creating an empty Dictionary
imageDict = {}

for imgName in roadImages:
    # read image
    imgName = str(imgName)
    imgName = imgName[2:]
    imgName = imgName[:-2]
    print(imgName)

    # read segmented for labels
    imgLabs = (cv2.imread(os.path.join(labeledImagesPath, str(imgName) + labelsExt), cv2.IMREAD_COLOR))

    for y in range(height):
        for x in range(width):

            # pixel values from segmented "parallel" images
            pixelLab = imgLabs[y, x]
            blueLab = (pixelLab[0])
            greenLab = (pixelLab[1])
            redLab = (pixelLab[2])
            if redLab == 128 and greenLab == 128 and blueLab == 128:
                countBareSoil += 1

    imageDict[imgName] = countBareSoil
    countBareSoil = 0

sort_roads = sorted(imageDict.items(), key=lambda x: x[1], reverse=True)

print(sort_roads[:5])

