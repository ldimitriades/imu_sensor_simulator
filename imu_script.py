import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt


##define constants
earth_ang_vel = np.array([0,0,7.292115])  #rad/sec

#helper functions
#create data loading function
def data_import(filename, file_type):
	#add all flexibility in input here, not sure if time will be included, if it is as keyed data or just another column, can adjust code then
	if file_type == 'csv_no_header':
		df = pd.read_csv(filename, header=None)
	if file_type == 'csv_with_header':
		df = pd.read_csv(filename)
	if file_type == 'with_comments':
		#can replace comment argument with whatever leading character in file 
		df = pd.read_csv(filename, comment='#')
	time = df.loc[:,0].to_numpy()
	position = df.loc[:,1:3].to_numpy()
	velocity = df.loc[:,4:6].to_numpy()
	proper_accel = df.loc[:,7:9].to_numpy()
	proper_accel_grav = df.loc[:,10:12].to_numpy()
	attitude = df.loc[:,13:15].to_numpy()
	ang_velocity = df.loc[:,16:].to_numpy()
	return time, position, velocity, proper_accel, proper_accel_grav, attitude, ang_velocity


def plotter_3d(vector, save=False, filepath=''):
	ax = plt.axes(projection='3d')
	ax.scatter(vector[0], vector[1], vector[2])
	ax.set_xlabel('x measurement')
	ax.set_ylabel('y measurement')
	ax.set_zlabel('z measurement')
	ax.grid(True)
	plt.tight_layout()
	if save:
		plt.savefig(filepath)
	plt.show()


#create rotational matrix
def rotation_matrix(attitude, representation):
	if representation == 'euler':
		a,b,c = attitude[0], attitude[1], attitude[2]
		element_11, element_12, element_13 = np.cos(a)*np.cos(b), np.sin(a)*np.cos(c) + np.cos(a)*np.sin(b)*np.sin(c), np.sin(a)*np.sin(c) - np.cos(a)*np.sin(b)*np.cos(c)
		element_21, element_22, element_23 = -np.sin(a)*np.cos(b), np.cos(a)*np.cos(c) - np.sin(a)*np.sin(b)*np.sin(c), np.cos(a)*np.sin(c) + np.sin(a)*np.sin(b)*np.cos(c)
		element_31, element_32, element_33 = np.sin(b), -np.cos(b)*np.sin(c), np.cos(b)*np.cos(c)
		return np.array([[element_11, element_12, element_13], [element_21, element_22, element_23], [element_31, element_32, element_33]])
	if representation == 'quaternions':
		q1,q2,q3,q4 = attitude[0], attitude[1], attitude[2], attitude[3]
		element_11, element_12, element_13 = 1-2*(q2**2 + q3**2), 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)
		element_21, element_22, element_23 = 2*(q2*q1 - q3*q4), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q1*q4)
		element_31, element_32, element_33 = 2*(q3*q1 + q2*q4), 2*(q3*q2 - q1*q4), 1 - 2*(q1**2 + q2**2)
		return np.array([[element_11, element_12, element_13], [element_21, element_22, element_23], [element_31, element_32, element_33]])
	if representation == 'DCM':
		return attitude.reshape(3,3)


def ang_acceleration(times, ang_velocity):
	"""function that takes in the angular velocities and times to estimate the angular acceleration using central differencing,
	can probably find a library with better method"""
	angular_accel = []
	#assuming equally spaced time samples
	time_step = ang_velocity[1] - ang_velocity[0]
	for i, omega in enumerate(ang_velocity):
		if i == 0:
			ang_accel = ang_velocity[i+1] - ang_velocity[i] / time_step 
		elif i == len(ang_velocity) - 1:
			ang_accel = ang_velocity[i] - ang_velocity[i-1] / time_step 
		else:
			ang_accel = ang_velocity[i+1] - ang_velocity[i-1] / 2*time_step
		angular_accel.append(ang_accel)
	return angular_accel



def imu_measurement(time, position, velocity, accceleration, attitude, ang_velocity,
					sensor_orientation,
					sensor_position, 
					bias_a=np.array([0,0,0]), 
					M_a=np.array([[0,0,0],[0,0,0],[0,0,0]]),
					bias_g=np.array([0,0,0]), 
					M_g=np.array([[0,0,0],[0,0,0],[0,0,0]])):
	"""the function takes in the state, accelerations, attitude, and angular velocity and models what the IMU would measure.
	Optionality has been included for the ECI(earth centered inertial), ECEF(earth-centered earth-fixed), and the S/C body frame.
	Bias, scale, and cross-coupled errors have been modeled as a potential input as well, defaulting to not being included.
	It is assumed that the accelerations are given in the same frame as velocity and position.
	If the sensor is place in orientation or origin not aligning with the body frame, then the correction needs to be made to model what the IMU would measure.
	It is assumed that the misaligned sensors are still orthonormal, so the previously derived rotation functions can be applied.
	The accelometer will feel forces due to being rotated relative to the body origin as well, which are modeled. 
	This requires the angular acceleration, which is not known. For this simple model, the angular velocity measurements
	to find the angular acceleration. 
	"""
	#need to account for 
						#1. inertial to body frame rotation 
						#2. tan./cen. effects of non-center of rotation mounting 
						#3. accel/gyro orientation misalignment(rotation matrix)
						
	if dataframe == 'ECI':
		a_body = rotation_matrix(attitude, representation) @ accceleration
		omega_body = rotation_matrix(attitude, representation) @ ang_velocity
	elif dataframe == 'ECEF':
		#rotation + cori/cen effects(need to import Earth angular velocity)
		a_pseudo = accceleration - 2*np.cross(earth_ang_vel, velocity) - np.cross(earth_ang_vel, np.cross(earth_ang_vel,position))
		a_body = rotation_matrix(attitude, representation) @ a_pseudo
		omega_body = rotation_matrix(attitude, representation) @ ang_velocity
	else:
		a_body = accceleration  

	#apply orientation and mounting error corrections
	angular_acceleration = ang_acceleration(time, ang_velocity)
	a_pos_corr = np.cross(a_body, r_misalignment) + np.cross(angular_acceleration, np.cross(angular_acceleration, r_misalignment))
	a = rotation_matrix(sensor_orientation, representation) @ a_pos_corr
	omega = rotation_matrix(sensor_orientation, representation) @ omega_body
	a_imu = (np.diag(3) + M_a)*a + bias_a
	omega_imu = (np.diag(3) + M_g)*omega_body + bias_g
	return a_imu, omega_imu




