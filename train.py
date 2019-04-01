import json
import os
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from scipy import stats

from time import time
from keras.callbacks import TensorBoard

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
run_type = 'IMU_Equal_AE'
# datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/processed_data_duplicate_removed_perUser_equal'
# run_type = 'User_Equal'

# datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/original_data'
# run_type = 'Original'

listing = os.listdir(datasetFolder)
# listing = sorted(listing, key=str.lower)
# print listing

Test_Accuracy = {}
TestALL_Accuracy = {}


def calulate_EER(y_actual, y_pred):
	fpr, tpr, threshold = roc_curve(y_actual, y_pred, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
	EER1 = fpr(np.nanargmin(np.absolute((fnr - fpr))))
	EER2 = fnr(np.nanargmin(np.absolute((fnr - fpr))))

	print('\nEER difference: {}').format(np.absolute(EER1-EER2))
	return EER1


def evaluate_model(trainX, trainy, testALLx, testALLy, testX, testy, NAME, repeats):
	epochs, batch_size, learning_rate = 40, 30, 0.001

	layer1 = Sequential()
	layer1.add(Convolution1D(len(col_names), 128, strides=2, activation='relu'))
	layer1.add(Convolution1D(len(col_names), 8, strides=2, activation='relu'))
	layer1.add(MaxPooling1D(pool_size=2))
	layer1.add(Dropout(0.2))
	layer1.add(Flatten())
	layer1.add(Dense(2, activation='softmax'))

	opt = keras.optimizers.adam(lr=learning_rate)

	layer1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	tensorboard = TensorBoard(log_dir='/media/adeen/Life/FYP/FYP_UPDATED/TENSORBOARD_LOGS/'+logs+'/'+run_number+'CNN128_batch30_{}'.format(time()))
	
	# layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(validation_X, validation_y), shuffle=True, callbacks = [tensorboard])
	layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks = [tensorboard])

	# layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)		# doesn't show training
	# layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True)				# shows training

	# evaluate model
	y1_pred = layer1.predict(testX)
	# yALL_pred = layer1.predict(testALLx)

	y1_pred_class = layer1.predict_classes(testX)
	# yALL_pred_class = layer1.predict_classes(testALLx)

	# show the inputs and predicted outputs
	print('\n                                      [pForged    pGenuine]')
	for i in range(len(testX)):
		# if(y1_pred[i] == 1 or testy[i] == 1):
		print('Actual=%s, Predicted=%s, Confidence=%s' % (testy[i, 1], y1_pred_class[i], y1_pred[i]))

	# EER = calulate_EER(testy[:,1], y1_pred_class)
	# print 'Actual EER = {}'.format(EER)
	# print '\n\n'

	# for i in range(len(testALLx)):
	# 	if(yALL_pred[i] == 1 or testALLy[i] == 1):
	# 		print("Actual=%s, Predicted=%s" % (testALLy[i], yALL_pred[i]))

	_, accuracy1 = layer1.evaluate(testX, testy, verbose=0)
	# _, accuracyALL = layer1.evaluate(testALLx, testALLy, verbose=0)

	model_name = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model.h5'
	layer1.save(model_name)

	model_weights = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model_weights.h5'
	layer1.save_weights(model_weights)

	if NAME == 'user45':
		model_architecture = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model_arch.json'
		architecture = layer1.to_json()
		
		with open(model_architecture, 'w') as f:
			json.dump(architecture, f)

	del layer1

	return accuracy1, 1
 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
	return m

 
# run an experiment
def run_experiment(trainX, trainy, testALLx, testALLy, testX, testy, NAME, repeats=1):
	scores1 = list()
	scores2 = list()

	for r in range(repeats):
		uno, dos = evaluate_model(trainX, trainy, testALLx, testALLy, testX, testy, NAME, r)
		score1 = uno * 100.0
		score2 = dos * 100.0
		print('>#%d: %.3f Test ---- %.3f TestALL\n' % (r+1, score1, score2))
		scores1.append(score1)
		scores2.append(score2)
	# summarize results
	print 'Test Set:'
	Test_Accuracy[NAME] = summarize_results(scores1)
	print '\n\nTestALL Set'
	TestALL_Accuracy[NAME] = summarize_results(scores2)


def pandafy(path, MAX_LENGTH):

	TEMP_ARR = list()
	for i in range(len(col_names)-1):
		TEMP_ARR.append(0)

	df = pd.read_csv(path, names = col_names)
	# print df.head()
	df.drop('time', axis = 1, inplace = True)
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = np.array(scaler.fit_transform(df))

	TEMP_ARR = np.zeros((MAX_LENGTH-len(scaled_data), len(TEMP_ARR)), dtype=float)
	scaled_data = np.append(scaled_data, TEMP_ARR, axis=0)
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

def load_data(NAME):

	X_tr = []       						# variable to store train dataset
	X_te = []								# variable to store test dataset

	count = 0
	
	MAX_LENGTH = maxlen(data_path)

	for subject in listing:
		data_path = datasetFolder+'/'+subject

		if os.path.isdir(os.path.join(data_path)):
			for csv in os.listdir(data_path):
				if os.path.isfile(os.path.join(data_path, csv)):
					if(feature in csv):
						path = data_path+'/'+csv
						scaled_data = pandafy(path)

						if subject != NAME:
							X_tr.append([scaled_data, [1, 0]])

						else:
							if count%2 == 0:
								X_te.append([scaled_data, [0, 1]])
							else:
								X_tr.append([scaled_data, [0, 1]])
							count+=1
							print subject, '  ', count

				elif os.path.isdir(os.path.join(data_path, csv)):
					for f in os.listdir(data_path+'/'+csv):
						if os.path.isfile(os.path.join(data_path+'/'+csv, f)):
							if(feature in f):
								path = data_path+'/'+csv+'/'+f
								scaled_data = pandafy(path)
						
								if subject != NAME:
									X_tr.append([scaled_data, [1, 0]])
									
								else:
									X_te.append([scaled_data, [1, 0]])
									# print subject,"_forged  ", count

	X_tr = np.array(X_tr)
	X_te = np.array(X_te)
	# print X_tr.shape, X_te.shape


	testX = np.array([i[0] for i in X_te])
	testy = np.array([i[1] for i in X_te])

	# put all the samples in testing, known and unknown
	trainX = np.array([i[0] for i in X_tr])
	trainy = np.array([i[1] for i in X_tr])

	testALLx = np.array(list(testX)+list(trainX))
	testALLy = np.array(list(testy)+list(trainy))

	run_experiment(trainX, trainy, testALLx, testALLy, testX, testy, NAME)





start_time = time()

# for i in listing:
# 	load_data(i)
load_data('user42')

print '\n\nTest Accuracy:\n', Test_Accuracy
print '\n\n'
print 'TestALL Accuracy:\n', TestALL_Accuracy
print '\n\n'

sum_test = []
sum_all_test = []

for i in Test_Accuracy.values():
	sum_test.append(float(i))
for i in TestALL_Accuracy.values():
	sum_all_test.append(float(i))

print 'Mean Test Accuracy: ', np.mean(np.array(sum_test))
print 'Mean TestALL Accuracy: ', np.mean(np.array(sum_all_test))

elapsed_time = (time() - start_time)/60

print '\nTime Elapsed = {} minutes\n'.format(elapsed_time)

with open('run_number.txt', 'w') as f:
	run_number = int(run_number)+1
	f.write(str(run_number))




# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Ori_logs/0oriSoftmax_all_mean.txt
# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Gyro_logs/0gyroSoftmax_all_mean.txt
# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Acc_logs/0accSoftmax_all_mean.txt
