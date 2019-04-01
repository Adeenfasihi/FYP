import json
import os
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from scipy import stats

from time import time
from sklearn.metrics import roc_curve


feature = '_ori_'
sensor = 'Orientation'
logs = 'Ori_logs'

# feature = '_acc_'
# sensor = 'Accelerometer'
# logs = 'Acc_logs'

# feature = '_gyr_'
# sensor = 'Gyroscope'
# logs = 'Gyro_logs'

run_number = '2'

with open('run_number.txt', 'r') as f:
	run_number = f.readline()

# Training data

col_names = []
if feature == '_ori_':
	col_names = ['time', 'x', 'y', 'z', 'w']
else:
	col_names = ['time', 'x', 'y', 'z']


# change dataset

datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/processed_data_equal_removed_equal_IMUs_length'
run_type = 'IMU_Equal'
# datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/processed_data_duplicate_removed_perUser_equal'
# run_type = 'User_Equal'

# datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/original_data'
# run_type = 'Original'

listing = os.listdir(datasetFolder)
# listing = sorted(listing, key=str.lower)
# print listing

model_folder = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor
user_models = os.listdir(model_folder)

Test_Accuracy = {}
FAR = {}
FRR = {}


def calulate_EER(y_actual, y_pred):
	fpr, tpr, threshold = roc_curve(y_actual, y_pred, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	far = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	frr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

	print(eer_threshold)
	return far,frr


def evaluate_model(testX, testy, NAME, repeats):
	# model_architecture = model_folder+'/user45/'+run_type+run_number+'_'+str(repeats)+'_model_arch.json'
	model_weights = model_folder+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model_weights.h5'
	
	# with open(model_architecture, 'r') as f:
		# layer1 = model_from_json(f.read())
	
	
	# model_name = model_folder+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model.h5'

	# try:
	# layer1 = load_model(model_name)
	# except:
	# 	print("Unsuccessful Load:\n%s" % model_name)
	# 	exit()


	epochs, batch_size, learning_rate = 40, 30, 0.001

	layer1 = Sequential()
	layer1.add(Convolution1D(len(col_names), 128, strides=2, activation='relu'))
	layer1.add(MaxPooling1D(pool_size=2))
	layer1.add(Dropout(0.2))
	layer1.add(Flatten())
	layer1.add(Dense(2, activation='softmax'))

	opt = keras.optimizers.adam(lr=learning_rate)
	layer1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	layer1.load_weights(model_weights, by_name=True)

	y1_pred = layer1.predict(testX)
	y1_pred_class = layer1.predict_classes(testX)

	print('\n                                      [pForged    pGenuine]')
	for i in range(len(testX)):
		print('Actual=%s, Predicted=%s, Confidence=%s' % (testy[i, 1], y1_pred_class[i], y1_pred[i]))

	far, frr = calulate_EER(testy[:,1], y1_pred_class)
	_, accuracy1 = layer1.evaluate(testX, testy, verbose=0)

	del layer1
	return accuracy1, far, frr
 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
	return m

 
# run an experiment
def run_experiment(testX, testy, NAME, repeats=1):
	scores1 = list()
	far = list()
	frr = list()
	for r in range(repeats):
		accuracy1, fa, fr = evaluate_model(testX, testy, NAME, r)
		score1 = accuracy1 * 100.0
		print('>#%d: %.3f Test\n' % (r+1, score1))
		scores1.append(score1)
		far.append(fa)
		frr.append(fr)

	# summarize results
	print 'Test Set:'
	Test_Accuracy[NAME] = summarize_results(scores1)
	print 'False ACCEPT Rate:'
	FAR[NAME] = summarize_results(far)
	print 'False REJECT Rate:'
	FRR[NAME] = summarize_results(frr)


def pandafy(path, MAX_LENGTH):

	TEMP_ARR = list()
	for i in range(len(col_names)-1):
		TEMP_ARR.append(0)

	df = pd.read_csv(path, names = col_names)
	df.drop('time', axis = 1, inplace = True)
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = np.array(scaler.fit_transform(df))

	TEMP_ARR = np.zeros((MAX_LENGTH-len(scaled_data), len(TEMP_ARR)), dtype=float)
	scaled_data = np.append(scaled_data, TEMP_ARR, axis=0)

	del TEMP_ARR
	del scaler
	del df

	return scaled_data

def maxlen(dir):
	length = 0
	max = length
	for root,dirs,files in os.walk(dir):
		for name in files:
			if ".py" not in name:
				length = len(open(os.path.join(root, name)).readlines())
				if max < length:
					max = length
					# print max, os.path.join(root, name), '\n'
	return max

def load_data(NAME):
	X_te = []								# variable to store test dataset

	count = 0

	data_path = datasetFolder+'/'+NAME
	MAX_LENGTH = maxlen(data_path)

	if os.path.isdir(os.path.join(data_path)):
		for csv in os.listdir(data_path):

			# load genuine signatures
			if os.path.isfile(os.path.join(data_path, csv)):
				if(feature in csv):
					path = data_path+'/'+csv

					if count%2 == 0:
						scaled_data = pandafy(path, MAX_LENGTH)
						X_te.append([scaled_data, [0, 1]])
					
					count+=1
					print NAME, '  ', count

			# load forgeries
			elif os.path.isdir(os.path.join(data_path, csv)):
				for f in os.listdir(data_path+'/'+csv):
					if os.path.isfile(os.path.join(data_path+'/'+csv, f)):
						if(feature in f):
							path = data_path+'/'+csv+'/'+f
							scaled_data = pandafy(path, MAX_LENGTH)
							X_te.append([scaled_data, [1, 0]])
								# print NAME,"_forged  ", count

	X_te = np.array(X_te)

	testX = np.array([i[0] for i in X_te])
	testy = np.array([i[1] for i in X_te])

	run_experiment(testX, testy, NAME)



start_time = time()

for i in listing:
	load_data(i)
# load_data('user45')

print '\n\nTest Accuracy:\n', Test_Accuracy
print '\n\n'
print 'FAR:\n', FAR
print '\n\n'
print 'FRR:\n', FRR
print '\n\n'

sum_test = []
far_sum = []
frr_sum = []

for i in Test_Accuracy.values():
	sum_test.append(i)

for i in FAR.values():
	far_sum.append(i)

for i in FRR.values():
	frr_sum.append(i)

print 'Mean Test Accuracy: ', np.mean(np.array(sum_test))
print 'Mean FAR: ', np.mean(np.array(far_sum))
print 'Mean FRR: ', np.mean(np.array(frr_sum))

elapsed_time = (time() - start_time)/60

print '\nTime Elapsed = {} minutes\n'.format(elapsed_time)




# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Ori_logs/0oriSoftmax_all_mean.txt
# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Gyro_logs/0gyroSoftmax_all_mean.txt
# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Acc_logs/0accSoftmax_all_mean.txt
