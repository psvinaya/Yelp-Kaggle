import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)

train_photos = pd.read_csv('../input/train_photo_to_biz_ids.csv')

import os
print(''.join([str(train_photos.photo_id[0]),'.jpg']))

from PIL import Image
im = Image.open(os.path.join('../input/','train_photos',''.join([str(train_photos.photo_id[0]),'.jpg'])))
plt.imshow(im)

train_attr = pd.read_csv('../input/train.csv')
train_attr['labels_list'] = train_attr['labels'].str.split(' ')
# find all the restaurants that are expensive (label=4)
train_attr['is_expensive'] = train_attr['labels'].str.contains('4')
expensive_businesses = train_attr[train_attr.is_expensive==True].business_id.tolist()
expensive_photos = train_photos[train_photos.business_id.isin(expensive_businesses)].photo_id.tolist()

