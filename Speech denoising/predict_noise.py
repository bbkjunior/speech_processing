import pickle
import argparse
import numpy as np
import os
import csv

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-n', '--npy_mel', type=str, help='normalized log-mel-spectrogramm in npy format')
parser.add_argument('-p', '--print_res', help='print prediction?', action="store_true")
parser.add_argument('-d', '--predict_for_files_in_directory',type=str, help='get directory which contains folders with npy files, make predictions for all files and otput results as csv')
args = parser.parse_args()

def predict_by_mel(mel, model,debug = False):
	if (debug): print("mel size", mel.shape)
	predictions = []
	for i in range(mel.shape[0]//80 + 1):
		if (80*(i+1)-1 < mel.shape[0]):
			if (debug): print("current size", 80*i, 80*(i+1)-1)
				
			ixgrid = np.ix_(range(80*i, 80*(i+1)), range(80))
			sample = mel[ixgrid]
			sample = sample.flatten()
			
			current_prediction = model.predict(sample.reshape(1, -1))
			
			predictions.append(current_prediction)
			
			#if (debug): print(sample.shape)
			if (debug): print(current_prediction)
		elif(80 - (80*(i+1)-1 - mel.shape[0]) > 30): 
			
			if (debug): print("current size", 80*i, 80*(i+1)-1)
			ixgrid = np.ix_(range(80*i, mel.shape[0]), range(80))
			if (debug): print(mel[ixgrid].shape)				
			pad_matrix = np.pad(mel[ixgrid], ((0, 80 - mel[ixgrid].shape[0]), (0, 0)), 'reflect')
			pad_matrix = pad_matrix.flatten()  
			if (debug): print(pad_matrix.shape) 
			current_prediction = model.predict(pad_matrix.reshape(1, -1))	
			predictions.append(current_prediction)
		else:
			if (debug): print("not enough real data left", 80 - (80*(i+1)-1 - mel.shape[0]))
	av_predictions = np.mean(predictions) 
	if (debug) : print(predictions)	  
	if(av_predictions > 0.5):
		final_prediction = 1
	else:
		final_prediction = 0
		 
	return final_prediction


#print(type(mel),mel.shape )
rf_model = pickle.load(open('rf.sav', 'rb'))

if(args.npy_mel):
	mel = np.load(args.npy_mel)
	
	prediction = predict_by_mel(mel, rf_model)

	if(args.print_res):
		if(prediction == 0):
			print("This spectrogramm is clean")
		elif(prediction == 1):
			print("This spectrogramm is noisy")
			
			
if(args.predict_for_files_in_directory):
	predicted_values = []
	for folder in os.listdir(args.predict_for_files_in_directory):
		print(folder)
		for file in os.listdir(os.path.join(args.predict_for_files_in_directory, folder)):
			mel = np.load(os.path.join(args.predict_for_files_in_directory, folder,file))
			prediction = predict_by_mel(mel, rf_model)
			predicted_values.append(prediction)

	with open('predicted_values.csv', 'w') as f:
		for val in predicted_values:
			f.write("%d\n"%(val))