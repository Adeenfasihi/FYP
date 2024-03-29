import json
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.optimizers import adam

from scipy import stats

from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint
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
MAX_LENGTH = 1350
# datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/processed_data_duplicate_removed_perUser_equal'
# run_type = 'User_Equal'

# datasetFolder = '/media/adeen/Life/FYP/FYP_UPDATED/data/original_data'
# run_type = 'Original'
# MAX_LENGTH = 2700

listing = os.listdir(datasetFolder)
listing = sorted(listing, key=str.lower)
# print listing

Test_Accuracy = {}
FAR = {}
FRR = {}
BEST_MODEL = {}


def calulate_EER(y_actual, y_pred):
	fpr, tpr, threshold = roc_curve(y_actual, y_pred, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	far = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	frr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

	# print(eer_threshold)
	return far,frr


def create_model():
	sequence_length = MAX_LENGTH
	features = len(col_names)-1

	layer1 = Sequential()
	layer1.add(Convolution1D(128, 4, strides=1, activation='relu', input_shape=(sequence_length, features)))
	layer1.add(MaxPooling1D(pool_size=2))
	layer1.add(Dropout(0.2))
	layer1.add(Flatten())
	layer1.add(Dense(2, activation='softmax'))

	return layer1


def evaluate_model(trainX, trainy, testX, testy, NAME, repeats):

	model_path = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor+'/'+NAME+'/'
	# model_ckpt = model_path+run_type+run_number+'_'+str(repeats)+'_model.ckpt'
	# ckpt_callback = ModelCheckpoint(model_ckpt, save_weights_only=True, verbose=0)
	# tb_callback = TensorBoard(log_dir='/media/adeen/Life/FYP/FYP_UPDATED/TENSORBOARD_LOGS/'+logs+'/'+NAME+'_'+run_type+run_number+'_'+str(repeats)+'CNN128_batch30_{}'.format(time()))
	# early_stop = keras.callbacks.EarlyStopping(patience=12, verbose=1)
	# reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1,patience=5, min_lt=0.00001,verbose=1)
	
	layer1 = create_model()

	epochs, batch_size, learning_rate = 40, 30, 0.001
	opt = adam(lr=learning_rate)
	layer1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	# layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(validation_X, validation_y), shuffle=True, callbacks = [tb_callback])
	# hist = layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=[ckpt_callback])

	# layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)		# doesn't show training
	hist = layer1.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)				# shows training
	
	model_weights = model_path+run_type+run_number+'_'+str(repeats)+'_model_weights.h5'
	layer1.save_weights(model_weights)

	# if NAME == 'user45':
	# 	model_architecture = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model_arch.json'
	# 	architecture = layer1.to_json()
		
	# 	with open(model_architecture, 'w') as f:
	# 		json.dump(architecture, f)
	


	# evaluate model
	y1_pred = layer1.predict(testX)
	y1_pred_class = layer1.predict_classes(testX)

	# show the inputs and predicted outputs
	print('\n                                      [pForged    pGenuine]')
	for i in range(len(testX)):
		# if(y1_pred[i] == 1 or testy[i] == 1):
		print('Actual=%s, Predicted=%s, Confidence=%s' % (testy[i, 1], y1_pred_class[i], y1_pred[i]))

	
	far, frr = calulate_EER(testy[:,1], y1_pred_class)
	_, accuracy1 = layer1.evaluate(testX, testy, verbose=0)

	# layer1.summary()

	# model_name = '/media/adeen/Life/FYP/FYP_UPDATED/MODELS/'+sensor+'/'+NAME+'/'+run_type+run_number+'_'+str(repeats)+'_model.h5'
	# layer1.save(model_name)


	del layer1
	return accuracy1, far, frr, model_path
 

# run an experiment
def run_experiment(trainX, trainy, testX, testy, NAME, repeats=3):
	best_acc = 0
	best_fa = 0
	best_fr = 0
	best_mo = ''
	mo_path = ''

	for r in range(repeats):
		accuracy1, fa, fr, bp = evaluate_model(trainX, trainy, testX, testy, NAME, r)
		
		if accuracy1 > best_acc:
			best_acc = accuracy1
			best_fa = fa
			best_fr = fr
			best_mo = run_type+run_number+'_'+str(r)+'_model_weights.h5'
			mo_path = bp

		print('Repeat>#%d: %.3f Test\n' % (r+1, accuracy1*100.0))

	# summarize results
	Test_Accuracy[NAME] = best_acc*100.0
	FAR[NAME] = best_fa*100.0
	FRR[NAME] = best_fr*100.0
	BEST_MODEL[NAME] = best_mo

	with open(mo_path+'best_model.txt', 'w') as f:
		f.write(str(best_mo))
		f.close()


def pandafy(path):

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


def load_data(NAME):

	X_tr = []       						# variable to store train dataset
	X_te = []								# variable to store test dataset

	count = 0
	
	# save model = ['user13_0', 'user15_fail', 'user16_2', 'user17_2','user43_0','user42_fail', 'user7_1_2', 'user2_fail', 'user36_1_2', 'user32_1_2','user38_1','user21_2','user23_0_1','user28_0_1','user24_0','user33_0_1','user20_0']
	# asd = ['user13', 'user15', 'user16', 'user17','user43','user42', 'user7', 'user2', 'user36', 'user32','user38','user21','user23','user28','user24','user33','user20']
	
	# if NAME not in asd:
	# 	return

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
							# print subject, '_', count

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
									# print 'forge_',data_path, f

	X_tr = np.array(X_tr)
	X_te = np.array(X_te)
	# print X_tr.shape, X_te.shape

	testX = np.array([i[0] for i in X_te])
	testy = np.array([i[1] for i in X_te])

	trainX = np.array([i[0] for i in X_tr])
	trainy = np.array([i[1] for i in X_tr])

	run_experiment(trainX, trainy, testX, testy, NAME)


start_time = time()

# for i in listing:
# 	load_data(i)
load_data('user2')

print '\n\nTest Accuracy:\n', Test_Accuracy
print '\n\n'
print 'FAR:\n', FAR
print '\n\n'
print 'FRR:\n', FRR
print '\n\n'
print 'Best Model\n:', BEST_MODEL
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

with open('run_number.txt', 'w') as f:
	run_number = int(run_number)+1
	f.write(str(run_number))
	f.close()


# python train_EER.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Ori_logs/GRU_Log0.txt
# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Gyro_logs/0gyroSoftmax_all_mean.txt
# python train.py |& tee /media/adeen/Life/FYP/FYP_UPDATED/TEXT_LOGS/Acc_logs/0accSoftmax_all_mean.txt
