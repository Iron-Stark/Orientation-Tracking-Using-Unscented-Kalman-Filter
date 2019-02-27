#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an extended kalman filter

from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
import itertools
import math
import Quaternion
import ukf
import warnings
import timeit

# def findAccelerometerSensitivity(ax,ay,az):
# 	return  np.sqrt(ax*ax+ay*ay+az*az)/(9.8*1023)
#
# def findBias(x, y, z, scalefactor, results):
# 	bias = np.zeros((3,))
# 	bias[0] = (x*scalefactor - results[0])/scalefactor
# 	bias[1] = (y*scalefactor - results[1])/scalefactor
# 	bias[2] = (z*scalefactor - results[2])/scalefactor
# 	return bias
# Check for singular
def rottoeuler(r):
	norm = np.sqrt(r[2,1]**2 + r[2,2]**2)
	roll_x = np.arctan(r[2,1]/r[2,2])
	pitch_y = np.arctan(-r[2,0]/norm)
	yaw_z = np.arctan(r[1,0]/r[0,0])
	return roll_x, pitch_y, yaw_z

def estimate_rot(data_num=6):
	#your code goes here
	imu_raw = sio.loadmat('imu/imuRaw'+str(data_num)+'.mat')
	imu_vals = imu_raw['vals']
	imu_ts = imu_raw['ts'].T
	imu = imu_vals#[:,36:]
	vicon_raw = sio.loadmat('vicon/viconRot'+str(data_num)+'.mat')
	vicon_rot = vicon_raw['rots']
	vicon_ts = vicon_raw['ts'].T
	vicon = vicon_rot
	vicon = vicon#[:,:,:-65]
	ax = -1*imu[0,:]
	ay = -1*imu[1,:]
	az = imu[2,:]
	acc = np.array([ax, ay, az]).T
	wx = imu[4,:]
	wy = imu[5,:]
	wz = imu[3,:]
	omega = np.array([wx, wy, wz]).T
	scale_acc = 3300/1023/330.0
	bias_acc = np.mean(acc[:50], axis = 0) - (np.array([0,0,1])/scale_acc)
	if data_num == 3:
		bias_acc = acc[0] - (np.array([0,0,1])/scale_acc)
	acc = (acc-bias_acc)*scale_acc
	scale_omega = 3300/1023/3.33
	omega = np.array([wx, wy, wz]).T
	#bias_omega = np.mean(omega[:bias], axis = 0)
	if data_num == 3:
		bias_omega = np.mean(omega[:250], axis = 0)
	bias_omega = np.mean(omega[:300], axis = 0)
	omega = (omega-bias_omega)*scale_omega*(np.pi/180)
	vicon_rolls = np.zeros((vicon.shape[2],))
	vicon_pitches = np.zeros((vicon.shape[2],))
	vicon_yaws = np.zeros((vicon.shape[2],))
	for i in range(vicon.shape[2]):
		vicon_rolls[i], vicon_pitches[i], vicon_yaws[i] = rottoeuler(vicon[:,:,i])
	previous_state_x = np.array([1,0,0,0])
	P = np.eye(3,3)
	Q = 8*np.eye(3,3)
	if data_num==3:
		Q = 50*np.eye(3,3)
		Q[2,2] = 2
		Q[1,2] = 100
	R = 8*np.eye(3,3)
	# print(bias_acc)
	predictions = np.zeros((3,3,ax.shape[0]))
	#start = timeit.default_timer()
	for i in range(ax.shape[0]-200):
		omegai = omega[i]
		W = ukf.sigma_points(P,Q)
		X = ukf.WtoX(W, previous_state_x)
		if i==ax.shape[0]-1:
			Y = ukf.XtoY(X,0.01, omegai)
		else:
			Y = ukf.XtoY(X,imu_ts[i+1]-imu_ts[i], omegai)
		# print(Y)
		ymean, W = ukf.meanY(Y, previous_state_x)
		#print(ymean)
		pk = ukf.calculatepk(W)
		Z = ukf.YtoZ(Y)
		zmean  = np.mean(Z, axis = 0)
		zmean = zmean/np.linalg.norm(zmean)
		v = ukf.innovation(zmean, acc[i])
		pzz = ukf.pzz(Z, zmean)
		pvv = ukf.pvv(pzz, R)
		pxz = ukf.pxz(W, Z, zmean)
		kk = ukf.kalman_gain(pxz, pvv)
		previous_state_x = ukf.update_state(pk,kk,v,ymean)
		P = ukf.update_cov(pk, kk, pvv)
		# if(data_num == 3 and i>3000):
		# 	break
		predictions[:,:,i] = Quaternion.quattorot(previous_state_x)
	#stop = timeit.default_timer()
	#print("Time: ",stop-start)
	prediction_rolls = np.zeros((predictions.shape[2],))
	prediction_pitches = np.zeros((predictions.shape[2],))
	prediction_yaws = np.zeros((predictions.shape[2],))

	for i in range(predictions.shape[2]):
		prediction_rolls[i], prediction_pitches[i], prediction_yaws[i] = rottoeuler(predictions[:,:,i])
	prediction_rolls = prediction_rolls.reshape((prediction_rolls.shape[0],))
	prediction_pitches = prediction_pitches.reshape((prediction_pitches.shape[0],))
	prediction_yaws = prediction_yaws.reshape((prediction_yaws.shape[0],))


	fig  = plt.figure()
	plt.subplot(3,1,1)


	plt.plot(imu_ts,prediction_rolls,'-r', label = 'Predicted')
	plt.plot(vicon_ts,vicon_rolls,'-b', label = 'Vicon')
	plt.legend(loc='best')
	plt.subplot(3,1,2)
	plt.plot(imu_ts,prediction_pitches,'-r', label = 'Predicted')
	plt.plot(vicon_ts,vicon_pitches,'-b', label = 'Vicon')
	plt.legend(loc='best')
	plt.subplot(3,1,3)
	plt.plot(imu_ts,prediction_yaws,'-r', label = 'Predicted')
	plt.plot(vicon_ts,vicon_yaws,'-b', label = 'Vicon')
	plt.legend(loc='best')
	plt.show()

if __name__ == "__main__":
	for i in range(1,4):
		estimate_rot(i)
