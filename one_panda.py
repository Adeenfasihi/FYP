import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import MaxAbsScaler

def scl(scaler):
	scaled_data = np.array(scaler.fit_transform(df))
	print(df.head())
	print(scaled_data[:4,:4])

	print("\n\n")

	# print(df.tail())
	# print(scaled_data[410:,:4])

	# print("\n\n")
	return scaled_data

def maxlen(dir):
	count = 0
	max = count
	for root,dirs,files in os.walk(dir):
		for name in files:
			if ".py" not in name:
				count = len(open(os.path.join(root, name)).readlines())
				if max < count:
					max = count
					print max, os.path.join(root, name), '\n'
	return max

col_names = ['time', 'x', 'y', 'z', 'w']

datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/processed_data_equal_removed_equal_IMUs_length'

df = pd.read_csv(datasetFolder+'/user2/19_ori_user2.csv', names = col_names)
df.drop('time', axis = 1, inplace = True)

scaled_data = scl(MinMaxScaler(feature_range=(0, 1)))
# scl(MaxAbsScaler())

MAX_LENGTH = maxlen(datasetFolder+'/user2')+2

TEMP_ARR = np.zeros((MAX_LENGTH-len(scaled_data), 4), dtype=float)
scaled_data = np.append(scaled_data, TEMP_ARR, axis=0)

print(scaled_data[410:,:4])

# datasetFolder = '/media/adeen/Life/FYP/DataCollection/DATA/3ori_data'
# listing = os.listdir(datasetFolder)

# print sorted(listing, key=str.lower)

