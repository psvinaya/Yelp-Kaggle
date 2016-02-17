import numpy as np # linear algebra
import pandas as pd
import collections
import util
import os
import random
import math


b2aCsvPath = '../input/train.csv'
p2bCsvPath = '../input/train_photo_to_biz_ids.csv'
processedInputDir = '../input/processed'
numAttributes=9

def writeToFileByAttribute(attrFilename,photosFilename, suffix):
	attr = util.readAttributesFromCSV(attrFilename)
	photos = util.readPhotosFromCSV(photosFilename)
	for i in xrange(numAttributes):
		b2a, p2b = util.getPhotosBusinessByAttribute(photos, attr, i)
		b2a.to_csv(processedInputDir+'/'+str(i)+'/'+suffix + '.csv', index=False)
		p2b.to_csv(processedInputDir+'/'+str(i)+'/'+suffix + '_photo_to_biz.csv', index=False)
		p2a = util.photoByAttributes(p2b, photos)
		p2a.to_csv(processedInputDir+'/'+str(i)+'/'+suffix + '_photo_to_attr.csv', index=False)


b2a = pd.read_csv(b2aCsvPath)
p2b = pd.read_csv(p2bCsvPath)
count = len(b2a.index)
valSize = int(math.floor(count/float(5)))
indexes = range(count)
chosen = random.sample(indexes, valSize)
chosenSet = set(chosen)
inputIndexes = list(set(indexes) - chosenSet)

if not os.path.exists(processedInputDir):
    os.makedirs(processedInputDir)
    for i in xrange(numAttributes):
		os.makedirs(processedInputDir+'/'+str(i))


#trainingSet:
trainBus, trainPhotos = util.getAttributesPhotos(b2a, p2b, inputIndexes)
valBus, valPhotos = util.getAttributesPhotos(b2a, p2b, chosen)

trainBusName = processedInputDir+'/trainBus.csv'
trainBus.to_csv(trainBusName,index=False)
trainPhotosName = processedInputDir+'/trainPhotos.csv'
trainPhotos.to_csv(trainPhotosName,index=False)
valBusName = processedInputDir+'/valBus.csv'
valBus.to_csv(valBusName,index=False)
valPhotosName = processedInputDir+'/valPhotos.csv'
valPhotos.to_csv(valPhotosName,index=False)

writeToFileByAttribute(trainBusName, trainPhotosName, 'train')
writeToFileByAttribute(valBusName, valPhotosName, 'val')




