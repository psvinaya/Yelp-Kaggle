import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from subprocess import check_output
from PIL import Image
import collections

inputDir = '../input/'
trainPhotosDir = 'train_photos'
trainCSVPath = '../input/train.csv'
testCSVPAth = '../input/sample_submission.csv'
trainPhotosCSVPath = '../input/train_photo_to_biz_ids.csv'
processedInputDir = '../input/processed'

labels = {
	0: 'good_for_lunch',
	1: 'good_for_dinner',
	2: 'takes_reservations',
	3: 'outdoor_seating',
	4: 'restaurant_is_expensive',
	5: 'has_alcohol',
	6: 'has_table_service',
	7: 'ambience_is_classy',
	8: 'good_for_kids'
}

def photoByAttributes(photosWithAttribute, allPhotos):
	newPhotos = allPhotos.copy(deep=True)
	newPhotos['attr'] = 0
	photoids = photosWithAttribute.photo_id.tolist()
	newPhotos['attr'][(newPhotos['photo_id'].isin(photoids))] = 1
	return newPhotos[['photo_id', 'attr']]

def getAttributesPhotos(train_attr, train_photos, ids):
	outBusinesses = train_attr.iloc[ids]
	bids = outBusinesses.business_id.tolist()
	outPhotos = getPhotosByBusiness(train_photos,bids)
	return outBusinesses, outPhotos

def getAccuracy(correct, predicted):
	if len(correct.index) != len(predicted.index):
		return -1
	incorrect = collections.Counter()
	for i in xrange(len(correct.index)):
		a = set(correct['labels'][0].split(" "))
		b = set(predicted['labels'][0].split(" "))
		a^=b
		for d in a:
			incorrect[int(d)]+=1
	return incorrect

def getBusinessByAttribute(train_attr, attribute):
	return train_attr[train_attr[attribute]==1]

def getPhotosBusinessByAttribute(train_photos, train_attr, attribute):
	businesses = getBusinessByAttribute(train_attr, attribute)
	photos = getPhotosByBusiness(train_photos, businesses.business_id.tolist())
	businesses = businesses[['business_id', attribute]]
	return businesses, photos

def getPhotosByBusiness(train_photos, businesses):
	return train_photos[train_photos.business_id.isin(businesses)]

def to_bool(s):
	return(pd.Series([1 if str(i) in str(s).split(' ') else 0 for i in range(9)]))

def readAttributesFromCSV(filename=trainCSVPath):
	train_attr = pd.read_csv(filename)
	Y = train_attr['labels'].apply(to_bool)
	Y['business_id'] = train_attr['business_id']
	return Y

def readPhotosFromCSV(filename=trainPhotosCSVPath):
	return pd.read_csv(trainPhotosCSVPath)

def viewImages(photos_to_show, count=10, l=5, w=2):
	plt.rcParams['figure.figsize'] = (10.0, 8.0)
	for x in xrange(count):
		plt.subplot(l, w, x+1)
        im = Image.open(os.path.join(inputDir,trainPhotosDir,''.join([str(photos_to_show[x]),'.jpg'])))
        plt.imshow(im)
        plt.axis('off')

