#TODO: DOCUMENT ENTIRE LIBRARY
#TODO: CREATE CUSTOM ERROR CALLS
#TODO: SET UP ALL DEFAULT AND SET_VAL TABLES TO CONTAIN DATATYPES
'''
set_vals = \
	{
		'item': [value, type]
	}
'''

import tensorflow as tf
from random import sample
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, fftconvolve

import copy

#	#	#	Backend Functions	#	#	#
class G2P_CORE():
	"""Core functions of creating a G2P network from scratch"""
	def systemID_input_gen_fcn(number_of_babbling_samples, pass_chance, max_in, min_in):
		"""Creates a uniform distribution of babbling activations while also ensuring values are between set maximum and minimum values and incorporating the ability to retain an activation level for multiple continuous samples"""
		gen_input = np.ones(number_of_babbling_samples,)*min_in

		for ii in range(1, number_of_babbling_samples):
			pass_rand = np.random.uniform(0,1,1)
			if pass_rand < pass_chance:
				gen_input[ii] = ((max_in-min_in)*np.random.uniform(0,1,1)) + min_in
			else:
				gen_input[ii] = gen_input[ii-1]
		return gen_input

	def inverse_mapping_func(limb_kinematics, limb_activations, logdir, downsampling_factor, max_epochs, hidden_layers, num_outputs, test_size=0.2):
		"""
		Creates a tensorflow model and fits kinematics as input and activations as output.
		
		Takes kinematics and activations as inputs, then splits that data into training and testing sets which can be downsampled. 
		The model is constructed based on the criteria given by the user at the construction of the G2P network and fitted with the kinematic training set as the inputs and 
		the activation testing set as the outputs and tested agains the test sets. The model is then saved.
		"""
		x_train_all, x_test_all, y_train_all, y_test_all = G2P_CORE.train_test_split(limb_kinematics, limb_activations, test_size) #TODO: add forgetfulness to this

		x_train=x_train_all[::downsampling_factor,:]
		x_test=x_test_all[::downsampling_factor,:]
		y_train=y_train_all[::downsampling_factor,:]
		y_test=y_test_all[::downsampling_factor,:]

		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir+"log")
		earlystopping_callback = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss', patience=5, verbose=1) #TODO: patience should be able to be input
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			logdir+"model/", monitor='val_loss', verbose=1, save_best_only=True)

		#defining the model structure
		model = tf.keras.Sequential()
		for layer_nodes in hidden_layers:
			#model.add(tf.keras.layers.Dense(layer_nodes, activation='sigmoid', kernel_initializer='glorot_uniform', input_shape=x_train.shape[1:])) #TODO: activation should be able to be input
			model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', kernel_initializer='glorot_uniform', input_shape=x_train.shape[1:])) #TODO: activation should be able to be input
		model.add(tf.keras.layers.Dense(num_outputs, activation='relu', kernel_initializer='glorot_uniform'))
		model.compile(optimizer=tf.keras.optimizers.Adam(0.0025), loss='mse', metrics=['mse']) #TODO: add customization to this

		history = \
			model.fit(
				x_train,
				y_train,
				epochs=max_epochs,
				validation_data = (x_test, y_test),
				callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback]
			)
		
		tf.keras.backend.clear_session()
		model = tf.keras.models.load_model(logdir+"model/", compile=False)
		return model

	### One layer down ###
	def train_test_split(x, y, test_size):
		"""Splits input data into two sets (training and testing)."""
		if x.shape[0]!=y.shape[0]:
			raise ValueError("number of input (x) and output (y) samples should be the same.") 
		samples_no=x.shape[0]
		test_indices=sample(range(samples_no),int(np.round(test_size*samples_no)))
		train_indices=list(set(range(samples_no))-set(test_indices))
		x_train=x[train_indices,:]
		x_test=x[test_indices,:]
		y_train=y[train_indices,:]

		y_test=y[test_indices,:]
		return x_train, x_test, y_train, y_test

#	#	#	Algorithmic Functions	#	#	#
class G2P():
	"""General-to-Particular class."""
	def __init__(self, dt, outputs, num_motors, num_sensors, **kwargs):
		"""Initializing General-to-Particular network"""
		
		# Creating default value table
		self.defaults = {
			'epochs': 20,
			'forget_block_size': 1000,
			'hidden_layers': [6],
			'seed': 0,
			'forgetful': 0,
			'logdir': './log/G2P/'
			#TODO: add assume_PVA (position, velocity, acceleration)
		}
		for key in self.defaults.keys():
			if key in kwargs:
				self.defaults[key] = kwargs[key]
		
		if 'hidden_nodes' in kwargs: 			# This takes care of the user deciding they want a single layer with a set number of nodes rather than multiple layers
			self.defaults['hidden_layers'] = [kwargs['hidden_nodes']]

		# Initialization Parameters
		self.dt = dt 							#timestep in seconds
		self.outputs = outputs 					#total number of outputs including motors being controlled
		self.sampling_frequency = 1/self.dt 	#timestep in hertz
		self.number_of_motors = num_motors 		#number of motors being controlled by this network
		self.number_of_sensors = num_sensors 	#assumption is that there will be sensors*3 inputs to the network
		
		np.random.seed(self.defaults['seed']) 	#setting the random seed

		# Saved data
		self.concatenated_kinematics = [] 		# All kinematics that the network will pull from for training
		self.concatenated_activations = [] 		# All activations that the network will pull from for training
		
	### Modification and retrieval of self variables ###

	def set_forget(self, forget_value):
		"""Allows the user to redefine the forgetfulness tag"""
		self.defaults['forgetful'] = bool(forget_value)
	
	def set_forget_block_size(self, block_size):
		"""Allows the user to redefine the size of the forget block"""
		self.defaults['forget_block_size'] = int(block_size)

	def get_babbling_inputs(self):
		"""Allows the user to retrieve the generated babbling activations"""
		if hasattr(self, 'babbling_input'):
			return self.babbling_input
		else:
			unbroken = True
			while unbroken:
				gen_W_defaults = input("It seems no babbling inputs have been generated; Would you like to generate inputs using defaults? (Y/N): ")
				if gen_W_defaults.lower() == 'y':
					self.generate_babbling()
					return self.babbling_input # generate new babbling activations if desired
				elif gen_W_defaults.lower() == 'n':
					print("No babbling inputs will be generated") # exit function if generation of babbling activations is not desired
					unbroken = False
				else:
					raise ValueError("Invalid input")

	def set_saved_activations_and_kinematics(self, activations, kinematics):
		"""Allows the user to redefine the concatenated activations and concatenated kinematics of the network with their own inputs"""
		if len(activations) == len(kinematics):
			if not isinstance(activations, np.ndarray):
				activations = np.array(activations)
			self.concatenated_activations = activations
			
			if not isinstance(kinematics, np.ndarray):
				kinematics = np.array(kinematics)
			
			if kinematics.shape[1] == self.number_of_sensors:
				kinematics = np.array(self.sensor_to_kinematics(kinematics))
				self.concatenated_kinematics = kinematics
			elif kinematics.shape[1] == self.number_of_sensors*3:
				self.concatenated_kinematics = kinematics
			else:
				raise ValueError("Size of kinematics list does not match kinematics based on number of sensors")
		else:
			raise ValueError("Activation list and kinematics list are different lengths")
	
	def add_activations_and_kinematics(self, activations, kinematics):
		"""Allows the user to add the input activations and kinematics to the concatenated activations and concatenated kinematics of the network"""
		if len(activations) == len(kinematics):
			if isinstance(activations, np.ndarray):
				activations = activations.tolist()
			if isinstance(kinematics, np.ndarray):
				kinematics = kinematics.tolist()
			if len(self.concatenated_activations) != 0:
				self.concatenated_activations = np.array(self.concatenated_activations.tolist() + activations)
			else:
				self.concatenated_activations = np.array(activations)
			
			if len(self.concatenated_kinematics) != 0:
				if len(kinematics[0])/3 == self.number_of_sensors: 
					self.concatenated_kinematics = np.array(self.concatenated_kinematics.tolist() + kinematics)
				elif len(kinematics[0]) == self.number_of_sensors:
					self.concatenated_kinematics = np.array(self.concatenated_kinematics.tolist() + self.sensor_to_kinematics(kinematics))
				else:
					raise ValueError("Size of kinematics list does not match kinematics based on number of sensors")
			else:
				if np.array(kinematics).shape[1] == self.number_of_sensors:
					kinematics = self.sensor_to_kinematics(kinematics)
					self.concatenated_kinematics = np.array(kinematics)
				elif np.array(kinematics).shape[1] == self.number_of_sensors*3:
					self.concatenated_kinematics = np.array(kinematics)
				else:
					raise ValueError("Size of kinematics list does not match kinematics based on number of sensors")
		else:
			raise ValueError("Activation list and kinematics list are different lengths")
	
	def get_saved_activations(self):
		"""Allows the user to retrieve all activations the network has saved"""
		return self.concatenated_activations

	def get_saved_kinematics(self, start=0, end=-1):
		"""
		Allows the user to retrieve all kinematics the network has saved beginning and ending anywhere in the array the user would like
		
		Params:
			start:	index the user desires to start from
			end:	index the user desired to end at
		"""
		if end==-1:
			end=len(self.concatenated_kinematics)
		return self.concatenated_kinematics[start:end]

	def get_saved_kinematics_shape(self):
		"""Returns the shape of the network's saved kinematics"""
		return self.concatenated_kinematics.shape
	
	### Utility functions ###

	def visualize_babbling_inputs(self):
		"""Creates a plot showing the babbling activations of each motor over time"""
		if hasattr(self, 'babbling_input'):
			plt.plot(np.linspace(0, len(self.babbling_input)*self.dt,len(self.babbling_input)), self.babbling_input)
			plt.xlabel('Time (s)')
			plt.ylabel('Activations amplitude')
			motors = []
			for i in range(self.number_of_motors):
				motors.append("Motor"+str(i))
			plt.legend(motors, loc='upper right')
			plt.show()
		else:
			raise AssertionError("Network has not generated any babbling inputs; Please generate inputs before visualizing")
	
	def generate_sin_cos_func(self, total_seconds, ordering, scales, **kwargs):
		"""
		Generates values corresponding to chosen trigonometric functions that are then scaled to given bounds
		
		Params:
			total_seconds:	total length of time in seconds that the output should last assuming a timestep of self.dt
			ordering:		string denoting the order of sine and cosine functions (ie. "SCC" for three inputs going sin, cos, cos) capital for forwards, lowercase for reverse
			scales:			minimum and maximum values for each function output

		"""
		'''Helper functions for chosing top and bottom values'''
		def select_top(li):
			abr = [abs(i) for i in li]
			j = abr.index(np.max(abr))
			return li[j]
		
		def select_bottom(li):
			abr = [abs(i) for i in li]
			j = abr.index(np.min(abr))
			return li[j]

		set_vals = \
			{
				'frequency': 1 #Frequency of the chosen trigonometric function(s) 
				#TODO: should this be able to be a list so the trig functions can have different frequencies?
				#TODO: should scales default to [-1, 1] and be an optional input?
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		sel_TB = []
		for i in range(1, len(scales), 2):
			sel = [scales[i-1], scales[i]]
			sel_TB.append([
				select_top(sel),		#top
				select_bottom(sel),		#bottom
				np.max(sel),			#max
				np.min(sel)				#min
			])

		samples = total_seconds/self.dt
		t = np.linspace(0, stop=total_seconds, num=int(samples))
		omega=2*np.pi*set_vals['frequency']
		
		q = []
		for cycle in ordering:
			if cycle == 'S':
				q.append(np.sin(omega*t))
			elif cycle == 's':
				q.append(-1*np.sin(omega*t))
			elif cycle == 'C':
				q.append(np.cos(omega*t))
			elif cycle == 'c':
				q.append(-1*np.cos(omega*t))
			else:
				raise ValueError("Unsupported function input")

		
		
		positions_sub = []
		for i in range(len(q)):
			set = sel_TB[i]
			positions_sub.append(
				((((-1*np.sign(set[0])*q[i])*(set[2]-set[3])/2)))+((((set[2]-set[3])/2))*np.sign(set[0]))+set[1]
			)
		positions = np.array(positions_sub).transpose()
		velocities = np.gradient(positions, self.dt, axis=0)
		accelerations = np.gradient(velocities, self.dt, axis=0)
		pre_kinematics = np.concatenate(
			(positions, velocities, accelerations), axis=1
		)
		kinematics = []
		for line in pre_kinematics:
			temp = []
			for i in range(len(q)):
				for j in range(i, len(line), len(q)):
					temp.append(line[j])
			kinematics.append(temp)
		kinematics = np.array(kinematics)
		
		return kinematics

	### Callable functions ###
	
	def generate_babbling(self, **kwargs):
		"""Generate activation values for each motor lasting the length of duration_in_seconds having a maximum value of max_value and a minimum value of min_value"""
		set_vals = {
				'pass_chance': 2*self.dt, #probability between 0 and 1 (assuming dt is 0.5 seconds or less) that the last activation stays unchanged #TODO: consider how this needs to be modified to accomidate larger dt values
				'duration_in_seconds': 2*60, 
				'max_value': 1, #TODO: should this be a list so that motors can have different maximum values?
				'min_value': 0, #TODO: should this be a list so that motors can have different minimum values?
				'return': False #determines if this function returns the activations it should generated to the user
			}
		for item in set_vals.keys():
			if item in kwargs:
				set_vals[item] = kwargs[item]
		
		num_samples = int(np.round(set_vals['duration_in_seconds']/self.dt))

		self.babbling_input = np.zeros((num_samples, self.number_of_motors))
		for motor_no in range(self.number_of_motors):
			self.babbling_input[:,motor_no] = G2P_CORE.systemID_input_gen_fcn(
				number_of_babbling_samples=num_samples,
				pass_chance=set_vals['pass_chance'],
				max_in=set_vals['max_value'],
				min_in=set_vals['min_value']
			)

		if set_vals['return']:
			return self.babbling_input
	
	def train(self, **kwargs):
		#TODO: set up so that epochs can change over time (more for babbling, less for refinements, and a different value for reinforcement learning refinements)
		"""Trains the network using the current values in concatenated_kinematics and concatenated_activations. Lasts for a set number of epochs and has the capability to downsample data as requested."""
		set_vals = {
			'epochs': self.defaults['epochs'],
			'downsampling_factor': 1,
			'logdir': self.defaults['logdir']
		}
		for item in set_vals.keys():
			if item in kwargs:
				set_vals[item] = kwargs[item]
		
		#normalize kinematics given to the network so that all inputs to the network are between 0 and 1
		normalized_kin = self.normalize_kinematics(self.concatenated_kinematics)

		model = G2P_CORE.inverse_mapping_func(
			normalized_kin,
			self.concatenated_activations,
			logdir = set_vals['logdir'],
			downsampling_factor=set_vals['downsampling_factor'],
			max_epochs=set_vals['epochs'],
			hidden_layers=self.defaults['hidden_layers'],
			num_outputs=self.outputs
		)
		self.model = model

		return model #return the network that was created and trained
	
	def predict(self, kin, **kwargs):
		"""Generate the activations for a single kinematic set"""
		set_vals = \
		{
			'min_val': 0,
			'max_val': 1
		}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#make sure kin is in the format we need
		if not isinstance(kin, np.ndarray):
			kin = np.array(kin)
		
		if kin.shape[0] == self.number_of_sensors or kin.shape[0] == self.number_of_sensors*3:
			if kin.shape[0] == self.number_of_sensors:
				raise ValueError("Kinematics require velocity and acceleration which were not provided and cannot be calculated")
		else:
			try:
				_ = kin.shape[1]
				raise ValueError("Size of kinematics list does not match kinematics based on number of sensors")
			except IndexError:
				raise ValueError("Expected one-dimensional array, but received a multi-dimensional array")
		
		#Normalize input kinematics to match the network
		normalized_kin = self.normalize_kin_to_network(np.array([kin.tolist()]))
		
		activations = self.model.predict(normalized_kin)
		activations = np.clip(activations, set_vals['min_val'], set_vals['max_val'])

		return activations
	
	def gen_activations_from_kinematics(self, kin_list, **kwargs): #this is for a full movement
		"""Generate the activations for a full movement"""
		set_vals = \
		{
			'min_val': 0,
			'max_val': 1
		}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#make sure kin is in the format we need
		if not isinstance(kin_list, np.ndarray):
			kin_list = np.array(kin_list)
		
		try:
			if kin_list.shape[1] == self.number_of_sensors or kin_list.shape[1] == self.number_of_sensors*3:
				if kin_list.shape[1] == self.number_of_sensors:
					kin_list = np.array(self.sensor_to_kinematics(kin_list))
			else:
				raise ValueError("Size of kinematics list does not match kinematics based on number of sensors")
		except IndexError:
			raise ValueError("Expected two-dimensional array, but received a one-dimensional array")
		
		#Normalize input kinematics to match the network
		normalized_kin = self.normalize_kin_to_network(kin_list)

		activations = self.model.predict(normalized_kin)
		activations = np.clip(activations, set_vals['min_val'], set_vals['max_val'])

		return activations

	def save(self, file_name):
		"""Save all values of the G2P network setup except for the actual network model to the given file link"""
		var_and_data_map = getattr(self, '__dict__')
		var_names = [name for name in list(var_and_data_map.keys()) if not(name in ['model'])]
		sub_dict = {k:var_and_data_map[k] for k in var_names}
		file = open(file_name+'.pkl', 'wb')
		pickle.dump(sub_dict, file)
		file.close()
	
	def load(self, file_name):
		"""Reload G2P network setup from given file link and then train network to reproduce the model"""
		file = open(file_name+'.pkl', 'rb')
		loaded_network = pickle.load(file)
		file.close()
		attr_name = list(loaded_network.keys())
		for i in range(len(attr_name)):
			setattr(self, attr_name[i], loaded_network[attr_name[i]])
		self.train()
		return self
	
	### One step down ###

	def sensor_to_kinematics(self, sensor_data):
		"""Take sensor values as input and concatenate with the first and second derivatives of it"""
		if not isinstance(sensor_data, np.ndarray):
			data = np.array(sensor_data)
		else:
			data = sensor_data
		positions = data[:,0:self.number_of_sensors]
		velocities = np.gradient(positions, self.dt, axis=0)
		accelerations = np.gradient(velocities, self.dt, axis=0)
		kinematics = np.concatenate((positions, velocities, accelerations), axis=1)
		return kinematics.tolist()
	
	def normalize_kinematics(self, kin):
		"""Take kinematics as input and return the normalization of all zeroth derivative values together, all first derivative values together, and all second derivative values together.
		
		This saves the needed values to modify new values to be scaled to the same normalization. 
		"""
		results = []
		mult = []
		add = []
		for i in range(3):
			temp = []
			for j in range(i,int(kin.shape[1]),3):
				temp.append(kin[:,j])
			temp = np.array(temp).transpose()
			results.append((temp-np.min(temp))/np.ptp(temp))
			mult.append(np.ptp(temp))
			add.append(np.min(temp))
		self.normalization_multipliers = mult
		self.normalization_additions = add
		final = []
		for i in range(int(kin.shape[1]/3)):
			for j in range(3):
				final.append(results[j][:,i])
		final = np.array(final).transpose()

		return final
	
	def normalize_kin_to_network(self, kin):
		"""Takes in kinematic values and scales them to match the normalization of the data that was used to train the network."""
		normalized_kin = np.zeros(kin.shape)
		for i in range(kin.shape[1]):
			normalized_kin[:,i] = \
				(kin[:,i]*self.normalization_multipliers[i%3]) \
				+self.normalization_additions[i%3]
		return normalized_kin


#	#	#	Templates	#	#	#
class G2P_object():
	"""Template class for a system using multiple G2P networks. 
	
	It is assumed that this will communicating with hardware and is designed to be extended to create
	a customized object for each robot type.
	"""
	#TODO: incorporate functions to reduce line count and repetition
	def __init__(self, limbs_in_order, motors_across_limbs, sensors_across_limbs, **kwargs):
		"""
		Initializes a G2P object
		
		Parameters:
			limbs_in_order:			list of limbs/limb names in the order they will be commanded/interacted in
			motors_across_limbs:	list of how many motors there are per limb in the order defined in limbs_in_order
			sensors_across_limbs:	list of how many sensors there are per limb in the order defined in limbs_in_order
		"""
		
		if len(motors_across_limbs) != len(limbs_in_order):
			raise ValueError("Number of limbs in motors_across_limbs does not match the number of limbs")
		if len(motors_across_limbs) != len(sensors_across_limbs):
			raise ValueError("Number of limbs in sensors_across_limbs does not match the number of limbs")
		
		self.default_vals = \
			{
				'ip':'',												#computer IP and port that will be used for inward communications
				'port':'',												#computer port that will be used for outwards communications
				'timestep_in_seconds':.25,								#amount of time in seconds that each step lasts for
				'outputs':np.sum(motors_across_limbs),					#number of outputs including motors such as: how many motors there are, values that will be modified during reinforcement learning, etc.
				'epochs': 20,											#base number of epochs each G2P network will train for
				'forget_block_size': 1000,								#number of timesteps to forget from the kinematics and activations of each limb
				'hidden_layers': [6],									#definition of how the hidden layers should be set up for the limbs' G2P networks #TODO: maybe set it up so that different limbs can have different hidden layer setups
				'seed': 0,												#value for the random seed
				'forgetful': False, 									#togglable value stating if the G2P networks created for the limbs should have forgetfulness
				'base_logdir': './log/G2P/', 							#directory starting point for the other files and directories that will be created and modified
				# TODO: add network setup (N networks, 1 network)
				'home_activations': [0.05]*np.sum(motors_across_limbs), #activations that are sent to the motors to establish a baseline for changes in position #TODO: maybe add a check to ensure this is the right length
				'homing_length': 2, 									#duration in seconds that the system should move to its home position before beginning to send activations
				'PI_P': 0.11,											#value for the proportional component of the PI controller
				'PI_I': 0.05 											#value for the integral component of the PI controller
				#TODO: consider setting this up so the user can define their own default_val items during initialization
			}
		for k in self.default_vals.keys():
			if k in kwargs:
				self.default_vals[k] = kwargs[k]

		#TODO: this does not handle the user giving 'hidden_nodes' in kwargs
		
		self.motors = motors_across_limbs
		self.sensors = sensors_across_limbs
		self.other_outputs = self.default_vals['outputs'] - np.sum(motors_across_limbs)

		self.limb_order = list(limbs_in_order)
		self.limbs = dict()
		# TODO: incorporate network setup (1 network vs multiple)

		# Create a G2P network for each limb
		i = 0
		for limb in self.limb_order:
			self.limbs[limb] = G2P(self.default_vals['timestep_in_seconds'], self.motors[i], self.motors[i], self.sensors[i], #using self.motors[i] for the outputs is a placeholder for the moment
			epochs=self.default_vals['epochs'], #TODO: this should be able to be passed as **kwargs
			forget_block_size=self.default_vals['forget_block_size'],
			hidden_layers=self.default_vals['hidden_layers'],
			seed=self.default_vals['seed'],
			forgetful=self.default_vals['forgetful'],
			logdir=self.default_vals['base_logdir'])
			i += 1
		
		self.limb_ranges = dict()
		i = 0
		for limb in self.limb_order:
			self.limb_ranges[limb] = [0]*self.sensors[i]*2
		
		self.rmse_errors = dict()

		# Refinement variables
		self.last_refinement = -1
		self.run_sections = dict()
		self.run_position_vector = dict()
		
		self.freq_scores = dict()
		for limb in self.limb_order:
			self.freq_scores[limb] = [-1]

		# Reinforcement variables
		self.prev_reward = np.array([0])
		self.best_reward_so_far = self.prev_reward
		self.all_rewards = []
		self.exploitation_run_no = 0
		self.best_features_so_far = dict()
	
	def blank_object(): #TODO: document this method
		return G2P_object([0], [0], [0])
	
	def save_object(self, folder_name, disregard_names=[]):
		"""Saves the system object as a directory containing a save file with all the internal values and a separate save file for each limb's network"""
		
		#determine if the save directory already exists and create one if it does not
		if not(folder_name[-1]=='/'):
			folder_name += '/'
		if not(os.path.isdir(folder_name)):
			os.makedirs(folder_name)
		
		#collect a list of values to save
		var_and_data_map = getattr(self, '__dict__')
		var_names = [name for name in list(var_and_data_map.keys()) if not(name in ['limbs'])and not(name in disregard_names)]
		sub_dict = {k:var_and_data_map[k] for k in var_names}
		
		#save the internal values
		file = open(folder_name+'self_values.sav', 'wb')
		pickle.dump(sub_dict, file)
		file.close()

		# save the limbs
		for limb in self.limb_order:
			self.limbs[limb].save(folder_name+'savedLimb_'+limb)
			
	def load_object(self, folder_name):
		"""Loads the system object from the directory provided and the contained save files"""
		
		#load internal values
		file = open(folder_name+'self_values.sav', 'rb')
		read_data = pickle.load(file)
		file.close()
		attr_name = list(read_data.keys())
		for i in range(len(attr_name)):
			setattr(self, attr_name[i], read_data[attr_name[i]])
		
		# Load the limbs and their networks
		self.limbs = dict()
		i = 0
		for limb in self.limb_order:
			temp = G2P(self.default_vals['timestep_in_seconds'], self.motors[i], self.motors[i], self.sensors[i],
			epochs=self.default_vals['epochs'],
			forget_block_size=self.default_vals['forget_block_size'],
			hidden_layers=self.default_vals['hidden_layers'],
			seed=self.default_vals['seed'],
			forgetful=self.default_vals['forgetful'],
			logdir=self.default_vals['base_logdir'])
			self.limbs[limb] = temp.load(folder_name+'savedLimb_'+limb) #TODO: making training the networks optional
			i += 1
		return self

	def connect_and_send(self, activations, test, **kwargs):
		"""An abstract method expected to be redefined by the user that will handle communication with their specific system"""
		raise NotImplementedError("Connection function has not been implemented, please extend this class and write to your own specifications")
	
	def set_home_activations(self, activations):
		"""Allows the user to redefine the home position activations"""
		if len(activations) == np.sum(self.motors):
			self.default_vals['home_activations'] = activations
		else:
			if isinstance(activations[0], list):
				self.default_vals['home_activations'] = activations[0]
			else:
				raise ValueError("There is a problem here") #TODO: put an actual error here
	
	def babble(self, **kwargs): #TODO: set up to allow for different motors having different min and max levels
		"""Generates and sends babbling activations for each defined limb within the system"""
		set_vals = \
			{
				'duration':120,												#Amount of time in seconds that the system will babble for
				'test': False,												#Debugging flag able to be used to signify not to send the activations to the actual system
				'min_value': 0.05,											#defines minimum activation level
				'max_value': 0.95,											#defines maxiumum activation level
				'pass_chance': 2*self.default_vals['timestep_in_seconds']	#defines the probability of staying at any given activation for another timestep
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#Generate babbling activations
		for limb in self.limbs.keys():
			self.limbs[limb].generate_babbling(duration_in_seconds=set_vals['duration'], min_value=set_vals['min_value'], max_value=set_vals['max_value'], pass_chance=set_vals['pass_chance'])
		separate_activations = []
		for limb in self.limb_order:
			separate_activations.append(self.limbs[limb].get_babbling_inputs().tolist())
		
		#convert from an activation list for each limb to one list that contains all activations at each timestep
		activations = []
		for i in range(len(separate_activations[0])):
			temp = []
			for j in range(len(separate_activations)):
				temp += separate_activations[j][i]
			activations.append(temp)
		self.babbling_activations = activations
		
		#send activations to system and receive kinematics back
		babbling_kinematics = self.connect_and_send(activations=activations, test=set_vals['test'], homing_length=self.default_vals['homing_length']) #TODO: insert kwargs somehow
		
		babbling_kinematics = np.array(babbling_kinematics)
		if hasattr(self, 'home_positions'):
			babbling_kinematics = babbling_kinematics - self.home_positions

		#define a label for the range of activations and kinematics that have been saved to the networks
		self.run_sections['babbling'] = [0, babbling_kinematics.shape[0]]

		#calculate the ranges for each sensor on each limb
		for i in range(len(self.limb_order)):
			temp = []
			for j in range(int(np.sum(self.sensors[:i])), int(np.sum(self.sensors[:i+1]))):
				temp.append(np.min(babbling_kinematics[:,j]))
				temp.append(np.max(babbling_kinematics[:,j]))
			self.limb_ranges[self.limb_order[i]] = temp.copy()
		
		print("Completed Babbing")
		print("Splitting Data")
		#split data for training
		self.babbling_kinematics = self.split_data(babbling_kinematics)
		print("Beginning training")
		#add data to each limb and train its network
		i = 0
		for limb in self.limb_order:
			self.limbs[limb].add_activations_and_kinematics(separate_activations[i], self.babbling_kinematics[limb])
			self.limbs[limb].train(logdir=self.default_vals['base_logdir']+'babbling/'+limb+'/')
			i += 1
		print("Completed training from babbling data")
	
	def babble_activations(self, **kwargs):
		#TODO: this can be removed by adding a check statement to the babble function
		# this just takes a snippet from "babble"
		set_vals = \
			{
				'duration':120,
				'test': False,
				'min_value': 0.05,
				'max_value': 0.95,
				'pass_chance': 2*self.default_vals['timestep_in_seconds']
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		for limb in self.limbs.keys():
			self.limbs[limb].generate_babbling(duration_in_seconds=set_vals['duration'], min_value=set_vals['min_value'], max_value=set_vals['max_value'], pass_chance=set_vals['pass_chance'])
		separate_activations = []
		for limb in self.limb_order:
			separate_activations.append(self.limbs[limb].get_babbling_inputs().tolist())
		activations = []
		for i in range(len(separate_activations[0])):
			temp = []
			for j in range(len(separate_activations)):
				temp += separate_activations[j][i]
			activations.append(temp)
		self.babbling_activations = activations

		s_activ = dict()
		i = 0
		for lis in separate_activations:
			s_activ[self.limb_order[i]] = lis
			i +=1
		return activations, s_activ
	
	def visualize_babbling_inputs(self):
		#TODO: checks to make sure this does not throw errors
		for limb in self.limb_order:
			self.limbs[limb].visualize_babbling_inputs()
	
	def visualize_babbling(self):
		#TODO: make sure this works
		return 0
	
	def add_training(self, activations, kinematics, **kwargs):
		"""Adds activations and kinematics to the limb network(s)"""
		set_vals = \
			{
				'no_training': False 		#flag to define if the user would like the networks to automatically train after adding the activations and kinematics or not
				#TODO: add a 'limbs' item to allow the user to do one limb at a time
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#this assumes that activations and kinematics are dictionaries
		# TODO: modify to allow for activations and kinematics to be lists 
		for limb in self.limb_order:
			self.limbs[limb].add_activations_and_kinematics(activations[limb], kinematics[limb])
			
			if not(set_vals['no_training']):
				self.limbs[limb].train()
		
		print("Successfully added activations and kinematics")
	
	def set_training_vals(self, activations, kinematics, **kwargs):
		"""Allows the user to redefine the activations and kinematics of the limb networks"""
		#TODO: this should also require update of self.run_sections
		set_vals = \
			{
				'no_training': False
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#this assumes that activations and kinematics are dictionaries
		# TODO: modify to allow for activations and kinematics to be lists 
		for limb in self.limb_order:
			self.limbs[limb].set_saved_activations_and_kinematics(activations[limb], kinematics[limb])
			
			if not(set_vals['no_training']):
				#self.limbs[limb].
				self.limbs[limb].train()
		
		print("Successfully set activations and kinematics")
	
	def split_data(self, data):
		"""Splits the data into the various limbs"""
		split_data = dict()
		data = np.array(data)
		i = 0
		j = 0
		for limb in self.limb_order:
			split_data[limb] = data[:,i:i+self.sensors[j]]
			i += self.sensors[j]
			j += 1
		return split_data
	
	def gen_desired_traj(self, **kwargs):
		"""Generates the trajectory that they system will attempt to replicate"""
		set_vals = \
		{
			'duration': 10,														#number of seconds the trajectory should last for
			'frequency': 1.25,													#frequency of the trigonometric functions
			'ordering': ["S"*self.sensors[i] for i in range(len(self.sensors))]	#ordering of the trigonometric functions broken down by limb
		}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		if not(type(set_vals['ordering']) == list):
			set_vals['ordering'] = [set_vals['ordering']]
		
		self.last_ordering = set_vals['ordering']

		trajectories = dict()
		i = 0
		for limb in self.limbs.keys():
			trajectories[limb] = self.limbs[limb].generate_sin_cos_func(set_vals['duration'], set_vals['ordering'][i], self.limb_ranges[limb], frequency=set_vals['frequency'])
			if len(set_vals['ordering']) > i+1:
				i += 1
		
		self.desired_traj = trajectories
	
	def conv_traj_to_activations(self):
		#TODO: check if this can be removed by adding a check statement to 'run_desired_traj'
		# this just takes a snippet from "run_desired_traj"
		if not hasattr(self, 'desired_traj'):
			choice = input("Currently no desired trajectories exist, would you like to generate some with default settings? (Y/N): ")
			if choice.lower() == 'y':
				self.gen_desired_traj()
			elif choice.lower() == 'n':
				print("Thank you for responding")
				return 0
			else:
				raise ValueError("Invalid input")
		
		limb_activations = dict()
		for limb in self.limbs.keys():
			limb_activations[limb] = self.limbs[limb].gen_activations_from_kinematics(self.desired_traj[limb])
		activations = []
		for i in range(len(limb_activations[self.limb_order[0]])):
			temp = []
			for limb in self.limb_order:
				temp += limb_activations[limb][i].tolist()
			activations.append(temp)
		
		return activations, limb_activations
	
	def run_desired_traj(self, **kwargs):
		"""
		Converts desired trajectory into kinematics then uses the limb networks to convert the kinematics to activations and send those to the target system. 
		
		This returns kinematics which can be used to further train the limb networks.
		"""
		set_vals = \
		{
			'test': False,
			'training_extension': False,
			'set_V_add': "A",
			'AoS_no_train': False,
			'passed_on(training)':{},
			'modified_traj': 0,
			'kin_to_run': self.desired_traj,
			'reward_feedback': False #TODO: this needs to be validated against prior G2P code and incorporated
			#TODO: add min and max values for clipping
		}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		if not hasattr(self, 'desired_traj'):
			choice = input("Currently no desired trajectories exist, would you like to generate some with default settings? (Y/N): ")
			if choice.lower() == 'y':
				self.gen_desired_traj()
			elif choice.lower() == 'n':
				print("Thank you for responding")
				return 0
			else:
				raise ValueError("Invalid input")
		
		## this is where the trajectories are run ##
		#first convert to activations
		limb_activations = dict()
		for limb in self.limbs.keys():
			if set_vals['modified_traj'] == 0:
				limb_activations[limb] = self.limbs[limb].gen_activations_from_kinematics(self.desired_traj[limb])
			elif set_vals['modified_traj'] == 1:
				limb_activations[limb] = self.limbs[limb].gen_activations_from_kinematics(self.modified_traj[limb])
			elif set_vals['modified_traj'] == 2:
				limb_activations[limb] = self.limbs[limb].gen_activations_from_kinematics(set_vals['kin_to_run'][limb])
			else:
				raise ValueError("Invalid modified_traj value")
		activations = []
		for i in range(len(limb_activations[self.limb_order[0]])):
			temp = []
			for limb in self.limb_order:
				#print(limb_activations[limb][i])
				temp += limb_activations[limb][i].tolist()
			activations.append(temp)

		#then send activations and receive kinematics back
		kin_back = np.array(self.connect_and_send(activations=activations, test=set_vals['test'], homing_length=self.default_vals['homing_length'])) #TODO: insert kwargs somehow
		kin_back = kin_back - self.home_positions
		
		if set_vals['training_extension']:
			split_data = self.split_data(kin_back)
			if set_vals['passed_on(training)']['logdir']:
				working_dir = set_vals['passed_on(training)']['logdir']
			else:
				working_dir = self.default_vals['base_logdir']
			for limb in self.limb_order:
				if set_vals['set_V_add'].upper() == "S":
					self.limbs[limb].set_saved_activations_and_kinematics(limb_activations[limb], split_data[limb])
				elif set_vals['set_V_add'].upper() == "A":
					self.limbs[limb].add_activations_and_kinematics(limb_activations[limb], split_data[limb])
				else:
					raise ValueError("Unsupported value for set_V_add argument")
			
				if not(set_vals['AoS_no_train']):
					set_vals['passed_on(training)']['logdir'] = working_dir + limb + '/'

					self.limbs[limb].train(**set_vals['passed_on(training)'])

		return limb_activations, kin_back
	
	def visualize_results(self, limb_name, startback=0): #TODO: this is wrong
		"""Create plots to display RMSE and limb trajectories to the user"""
		num_sensors = self.sensors[self.limb_order.index(limb_name)]
		if startback == 0:
			startback = self.last_refinement
		rmse_errors = np.zeros((self.last_refinement+1, num_sensors))
		for ref_no in range(self.last_refinement+1):
			rmse_errors[ref_no, :] = np.transpose(np.array(self.rmse_errors[ref_no][limb_name]))
		fig0, axs0 = plt.subplots(num_sensors, startback+1)
		for ref_no in range(self.last_refinement-startback, self.last_refinement+1, 1):
			for i in range(num_sensors):
				axs0[i, ref_no-(self.last_refinement-startback)].plot(self.limbs[limb_name].get_saved_kinematics(start=self.run_sections['ref_'+str(ref_no)][0], end=self.run_sections['ref_'+str(ref_no)][1])[:, i]) #get just the angle
				axs0[i, ref_no-(self.last_refinement-startback)].plot(self.desired_traj[limb_name][:, i*3])
				if i == 0:
					axs0[i,ref_no-(self.last_refinement-startback)].set_title('attempt no: '+str(ref_no))
				if ref_no == self.last_refinement-startback:
					axs0[i, ref_no-(self.last_refinement - startback)].set_ylabel('Joint '+str(i))
		
		# Hide x labels and tick labels for top plots and y ticks for right plots.
		for ax in axs0.flat:
			ax.label_outer()
		print('rmse errors for ' + limb_name +': ')
		print(rmse_errors)
		fig1, axs1 = plt.subplots(num_sensors, 1)
		for i in range(num_sensors):
			axs1[i].plot(rmse_errors[:,i])
			axs1[i].set_title('Joint '+str(i))
			axs1[i].set_ylabel('rmse error')
			if i == num_sensors-1:
				axs1[i].set_xlabel('attempt no')
		
		if len(self.freq_scores[limb_name]) > 1:
			fig2, axs2 = plt.subplots(1,1)
			axs2.plot(self.freq_scores[limb_name][:])
			axs2.set_title(str(limb_name))
			axs2.set_ylabel('freq_score')
			axs2.set_xlabel('attempt no')
		
		# Hide x labels and tick labels for top plots and y ticks for right plots.
		for ax in axs1.flat:
			ax.label_outer()
		plt.show()
		return rmse_errors

	### Refinement code ###
	def set_last_refinement(self, num_last_refinement):
		"""
		Allow the user to redefine what the last refinement was.
		
		This allows the user to essentially jump back to an earlier refinement.
		"""
		self.last_refinement = num_last_refinement
	
	def set_PI(self, P, I):
		"""Allows the user to redefine the P and I terms in the PI controller"""
		self.default_vals['PI_P'] = P
		self.default_vals['PI_I'] = I
	
	def gen_vel_compensation_term(self, sample_no, limb_name, refinement_num, prev_desired_kin, run_position_error_vector):
		"""Uses the errors in limb trajectory to calculate a velocity compensation term to correct the motion"""
		prev_desired_position_vector = \
			self.limbs[limb_name].get_saved_kinematics(
				start=self.run_sections["ref_"+str(refinement_num)][0], 
				end=self.run_sections["ref_"+str(refinement_num)][1])\
					[sample_no-1, :self.sensors[self.limb_order.index(limb_name)]]
		prev_position_vector = prev_desired_kin[sample_no-1, :self.sensors[self.limb_order.index(limb_name)]]
		
		position_error_vector = prev_desired_position_vector - prev_position_vector
		run_position_error_vector += position_error_vector * self.default_vals['timestep_in_seconds']
		velocity_compensation_term = (self.default_vals['PI_P']*position_error_vector) + (self.default_vals['PI_I']*run_position_error_vector)
		
		return velocity_compensation_term, run_position_error_vector
	
	def compensated_traj(self, refinement_num, comp_type): #TODO
		"""Creates a new trajectory that is the combination of the desired trajectory kinematics and the velocity compensation term"""
		compensated_traj = dict()
		run_pos_vector = dict()
		for limb in self.limb_order:
			run_pos_vector[limb] = [0]*self.sensors[self.limb_order.index(limb)]
			comp = []
			if comp_type == "PI":
				for sample in range(len(self.desired_traj[limb])):
					term, run_pos_vector[limb] = self.gen_vel_compensation_term(sample, limb, refinement_num, self.desired_traj[limb], run_pos_vector[limb])
					comp.append(term)
				compensated_traj[limb] = comp
			
			elif comp_type == "CORR":
				comp, _ = self.correlation_based_mod(limb_name=limb, reference_num=refinement_num) #TODO: this has more customizability options that are not being used
				compensated_traj[limb] = comp
			
			elif comp_type == "COMBO": #TODO: per joint perhaps
				comp, freq_score = self.correlation_based_mod(limb_name=limb, reference_num=refinement_num) #TODO: this has more customizability options that are not being used
				#if freq_score <= self.freq_scores[limb][-1]:
				if refinement_num > 3 and freq_score < self.freq_scores[limb][-1]:
					comp = []
					for sample in range(len(self.desired_traj[limb])):
						term, run_pos_vector[limb] = self.gen_vel_compensation_term(sample, limb, refinement_num, self.desired_traj[limb], run_pos_vector[limb])
						comp.append(term)
					compensated_traj[limb] = comp
				else:
					compensated_traj[limb] = comp
				self.freq_scores[limb].append(freq_score)
			else:
				raise ValueError("Unknown compensation type")
		
		return compensated_traj
	
	def modify_traj(self, modifications):
		"""Allows the user to make modifications to the velocity component of the desired trajectory kinematics"""
		mod_traj = dict()
		for limb in self.limb_order:
			for i in range(self.desired_traj[limb].shape[0]):
				temp = []
				for j in range(self.sensors[self.limb_order.index(limb)]*3):
					if j%3 == 1:
						temp.append(self.desired_traj[limb][i, j]+modifications[limb][i][int(j/3)])
					else:
						temp.append(self.desired_traj[limb][i][j])
				try:
					mod_traj[limb].append(temp)
				except KeyError:
					mod_traj[limb] = [temp]
		self.modified_traj = mod_traj
	
	def get_rmse(self, limb_name, reference_num):
		"""
		Allows the user to get the RMSE of a given limb for a given refinement

		Parameters:
			limb_name:		Name of the desired limb
			reference_num:	Number of the refinement the user desires to look at
		"""
		#TODO: this only looks at position (maybe add in a look at velocity and acceleration too)
		kinematics = self.limbs[limb_name].get_saved_kinematics(start=self.run_sections['ref_'+str(reference_num)][0], end=self.run_sections['ref_'+str(reference_num)][1])
		errors = []
		for i in range(self.sensors[self.limb_order.index(limb_name)]):
			errors.append((((kinematics[:,i]-self.desired_traj[limb_name][:,i])**2).mean(axis=0))**0.5) #TODO: is this even correct?
		return errors
	
	def correlation_based_mod(self, limb_name, reference_num, **kwargs): #TODO: add more customizability
		set_vals = \
		{
			'min': -1,
			'max': 1
		}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		def normalize(signal, a, b):
			# solving system of linear equations one can find the coefficients
			A = np.min(signal)
			B = np.max(signal)
			C = (a-b)/(A-B)
			k = (C*A - a)/C
			return (signal-k)*C
		
		kinematics = self.limbs[limb_name].get_saved_kinematics(start=self.run_sections['ref_'+str(reference_num)][0], end=self.run_sections['ref_'+str(reference_num)][1])
		freq_errors = normalize(fftconvolve(kinematics[:,:], self.desired_traj[limb_name][:,:], mode='same'), set_vals['min'], set_vals['max'])
		comparative_freq = (kinematics/freq_errors)
		comparative_freq = np.maximum(comparative_freq, -100)
		comparative_freq = np.minimum(comparative_freq, 100)
		comparative_freq = normalize(comparative_freq, -1, 1)
		comparative_freq = normalize((kinematics*comparative_freq)*freq_errors, -1, 1)
		kinematics_rectified = self.desired_traj[limb_name]*comparative_freq
		
		freq_score = abs(np.average((self.desired_traj[limb_name]/freq_errors)))
		return kinematics_rectified, freq_score


	def refine_networks(self, num_refinements, redefine_traj=False, **kwargs):
		"""
		Takes in desired number of refinements and runs the desired trajectory that many times and training the limb networks after each run.
		
		Optionally the user can also add velocity compensation to add PI control
		"""
		set_vals = \
			{
				'test': 0,
				'set_V_add': 'a',
				'continue_refinements':True,
				'working_logdir': self.default_vals['base_logdir'],
				'velocity_compensation': False,
				'traj_comp_type': 'PI'
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		# redefine_traj should be a dictionary containing what the user wants to send to gen_desired_traj
		if redefine_traj != False:
			self.gen_desired_traj(**redefine_traj)
		
		if not hasattr(self, 'desired_traj'):
			choice = input("Currently no desired trajectories exist, would you like to generate some with default settings? (Y/N): ")
			if choice.lower() == 'y':
				self.gen_desired_traj()
			elif choice.lower() == 'n':
				print("Thank you for responding")
				return 0
			else:
				raise ValueError("Invalid input")
		set_vals['passed_on(run_desired_traj)'] = \
			{'test':set_vals['test'], 
			'training_extension':True, 
			'AoS_no_train':False, 
			'set_V_add':set_vals['set_V_add'],
			'velocity_compensation':set_vals['velocity_compensation'], 
			'passed_on(training)':{'logdir':'/'}
			#'max_value' = set_vals['max_value'],
			#'min_value' = set_vals['min_value'] #TODO: this
			}

		if set_vals['continue_refinements']:
			for i in range(self.last_refinement+1, self.last_refinement + num_refinements + 1):
				self.rmse_errors[i] = dict()
				set_vals['passed_on(run_desired_traj)']['passed_on(training)']['logdir'] = set_vals['working_logdir'] + 'refinement_'+str(i)+'/'

				if set_vals['velocity_compensation']:
					if self.last_refinement > -1:
						compensated_traj = self.compensated_traj(refinement_num=self.last_refinement, comp_type=set_vals['traj_comp_type'])
						self.modify_traj(compensated_traj)
						self.run_desired_traj(**set_vals['passed_on(run_desired_traj)'], modified_traj=1)
					else:
						self.run_desired_traj(**set_vals['passed_on(run_desired_traj)'])
				else:
					self.run_desired_traj(**set_vals['passed_on(run_desired_traj)'])
				self.last_refinement += 1
				
				last_item = self.limbs[self.limb_order[0]].get_saved_kinematics_shape()[0]
				self.run_sections['ref_'+str(self.last_refinement)] = [last_item-len(self.desired_traj[self.limb_order[0]]), last_item] #TODO: I don't think this works if they are trying to rewrite data

				for limb in self.limb_order:
					self.rmse_errors[i][limb] = self.get_rmse(limb, i)
				

		else: #TODO: there is a lot to do with this that I am just ignoring for the moment
			for i in range(num_refinements):
				#TODO: make this set the values instead of add on the first one
				set_vals['passed_on(run_desired_traj)']['passed_on(training)']['logdir'] = set_vals['working_logdir'] + 'refinement_'+str(i)+'/'
				self.run_desired_traj(**set_vals['passed_on(run_desired_traj)'])
				self.last_refinement = i

				last_item = self.limbs[self.limb_order[0]].get_saved_kinematics_shape()[0]
				self.run_sections['ref_'+str(self.last_refinement)] = [last_item-len(self.desired_traj[self.limb_order[0]]), last_item]


	### Reinformcement Code ###
	#TODO: add reinforcement learning

	def gen_features_func(reward_thresh, best_reward_so_far, **kwargs): #TODO: CONFIRM THIS WORKS AS INTENDED
		"""Generate the feature vector based on best_reward_so_far, else based on a uniform distribution"""
		set_vals = \
			{
				'feat_min': 0.4,
				'feat_max': 0.9,
			}
		
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		if "best_features_so_far" in kwargs:
			set_vals['best_features_so_far'] = kwargs['best_features_so_far']
		elif "feat_vec_length" in kwargs:
			set_vals['best_features_so_far'] = np.random.uniform(set_vals['feat_min'], set_vals['feat_max'], kwargs['feat_vec_length'])
		else:
			raise ValueError("Either best_features_so_far or feat_vec_length must be provided")
		
		if best_reward_so_far < reward_thresh:
			new_features = np.random.uniform(set_vals['feat_min'], set_vals['feat_max'], set_vals['best_features_so_far'].shape[0])
		else:
			sigma = np.max([(12-best_reward_so_far)/100, 0.01]) #TODO: should this be 12?
			new_features = np.zeros(set_vals['best_features_so_far'].shape[0],)
			for ijk in range(set_vals['best_features_so_far'].shape[0]):
				new_features[ijk] = np.random.normal(set_vals['best_features_so_far'][ijk], sigma)
			new_features = np.maximum(new_features, set_vals['feat_min']*np.ones(set_vals['best_features_so_far.shape'][0],))
			new_features = np.minimum(new_features, set_vals['feat_max']*np.ones(set_vals['best_features_so_far'].shape[0],))
		return new_features
	
	def feat_to_positions_func(self, limb, features, scales, **kwargs): #TODO: test this
		"""Convert the generated feature vector to a desired trajectory to follow"""
		#TODO: refactor this to make it shorter and more readable
		set_vals = \
			{
				'dt': self.default_vals['timestep_in_seconds'],
				'show': False,
				'cycle_duration_in_seconds': 1,
				'a':1
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		ranges = []
		offsets = []
		for i in range(0, len(scales), 2):
			ranges.append(scales[i+1]-scales[i])
			offsets.append((scales[i]+scales[i+1])/2)
		
		num_features = features.shape[0]
		single_feature_length = int(np.round((set_vals['cycle_duration_in_seconds']/num_features)/set_vals['dt']))
		feat_angles = np.linspace(0, 2*np.pi*(num_features/(num_features+1)), num_features)

		ordering = self.last_ordering[self.limb_order.index(limb)]
		raw_positions = []
		for cycle in ordering:
			if cycle == 'S':
				raw_positions.append(features*np.sin(feat_angles))
			elif cycle == 's':
				raw_positions.append(features*-1*np.sin(feat_angles))
			elif cycle == 'C':
				raw_positions.append(features*np.cos(feat_angles))
			elif cycle == 'c':
				raw_positions.append(features*-1*np.cos(feat_angles))
			else:
				raise ValueError("Unsupported function input")

		scaled_positions = []
		for i in range(len(raw_positions)):
			scaled_positions.append((raw_positions[i]*ranges[i])+offsets[i])
		
		extended_scaled_pos = []
		for pos in scaled_positions:
			extended_scaled_pos.append(pos + [pos[0]])
		
		extended_scaled_long_positions = []
		for _ in extended_scaled_pos:
			extended_scaled_long_positions.append(np.array([]))
		for ijk in range(features.shape[0]):
			for esp in range(len(extended_scaled_pos)):
				extended_scaled_long_positions.append(np.append(extended_scaled_long_positions[i], np.linspace(extended_scaled_pos[esp][ijk],extended_scaled_pos[esp][ijk+1],single_feature_length)))
		
		extended_scaled_long_triple_positions = []
		for pos in extended_scaled_long_positions:
			extended_scaled_long_triple_positions.append(np.concatenate([pos[:-1],pos[:-1],pos[:]]))
		
		fir_filter_len = int(np.round(single_feature_length/(1))) #TODO: figure out what this 1 is for and make it modular
		b = np.ones(fir_filter_len,)/fir_filter_len

		triple_positions_filtered = []
		for esltp in extended_scaled_long_triple_positions:
			triple_positions_filtered.append(filtfilt(b, set_vals['a'], esltp))
		
		filtered_positions = []
		for tpf in range(len(triple_positions_filtered)):
			filtered_positions.append(triple_positions_filtered[tpf][extended_scaled_long_positions[tpf].shape[0]:2*extended_scaled_long_positions[tpf].shape[0]-1])
		
		return filtered_positions
	
	def positions_to_kinematics_func(positions, dt): #TODO: test this
		"""convert positions to kinematics by taking the first and second derivatives"""
		#TODO: check if this can be done using another function that has already been defined
		kin_grad = []
		for item in positions:
			kin_grad.append([item])
			kin_grad.append([np.gradient(item)/dt])
			kin_grad.append([np.gradient(np.gradient(item)/dt)/dt])
		kinematics = np.transpose(kin_grad)
		return kinematics
	
	def movement_to_attempt_kinematics_func(move_kin, num_repetitions_per_run=10):
		"""Convert movement to attempt by repeating the movement a given number of times""" 
		#TODO: test this
		attempt_kinematics = np.matlib.repmat(move_kin, num_repetitions_per_run, 1)
		return attempt_kinematics

	def feat_to_attempt_single_limb(self, features, scales, **kwargs):
		"""Convert feature vector to kinematics"""
		set_vals = \
			{
				'timestep': self.default_vals['timestep_in_seconds'],
				'feat_show': False,
				'repetitions': 10
			}
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#Convert feature vector to desired trajectory
		positions_filtered = self.feat_to_positions_func(features, scales, dt=set_vals['timestep'], show=set_vals['feat_show'])

		#convert desired trajectory to kinematics
		limb_movement_kinematics = self.positions_to_kinematics_func(positions_filtered, dt=set_vals['timestep'])

		#repeat to create a full movement
		attempt_kinematics = self.movement_to_attempt_kinematics_func(move_kin=limb_movement_kinematics, num_repetitions_per_run=set_vals['repetitions'])

		return attempt_kinematics
	
	def save_features_func(self, logdir):
		"""Save best features that are found"""
		save_dir = logdir + 'features/'
		try:
			feature_file = open(save_dir+'best_features.pkl', 'wb')
		except:
			os.mkdir(save_dir)
			feature_file = open(save_dir+'best_features.pkl', 'wb')
		finally:
			pickle.dump(self.best_features_so_far, feature_file)

	def learn_to_move_func(self, **kwargs): #TODO: test this
		"""Use reinforcement learning to refine a feature vector for each limb and produce movements from cyclical functions"""
		#TODO: THIS ONLY ACCOUNTS FOR REFINEMENTS TO LIMB MOVEMENTS, MODIFY TO INCLUDE EXTERNAL KNOBS THAT CAN BE MANIPULATED
		#TODO: add cost function for reward
		set_vals = \
			{
				'reward_thresh': 6,
				'refinement': False,
				'exploitations': 15, #the number of runs that must be above the reward threshold before stopping
				'feat_show': False,
				'show_final': False, 
				'feature_vector_length': 30 #how many segments of the circle
			}
		
		for k in set_vals.keys():
			if k in kwargs:
				set_vals[k] = kwargs[k]
		
		#Handling of log_directory
		func_base_logdir = self.default_vals['base_logdir'] + '/reinforcement/'

		new_features = dict()
		attempt_kin = dict()

		for limb in self.limb_order: # Each limb is given their own feature vector
			new_features[limb] = self.gen_features_func(reward_thresh=set_vals['reward_thresh'], best_reward_so_far=self.best_reward_so_far, feat_vec_length=set_vals['feature_vector_length']) 
		self.best_features_so_far = copy.deepcopy(new_features)

		while self.exploitation_run_no <= 15:
			if self.best_reward_so_far > set_vals['reward_thresh']:
				self.exploitation_run_no += 1
			for limb in self.limb_order:
				new_features[limb] = self.gen_features_func(reward_thresh=set_vals['reward_thresh'], best_reward_so_far=self.best_reward_so_far, best_features_so_far=self.best_features_so_far[limb])
				attempt_kin[limb] = self.feat_to_attempt_kin_single_limb_func(features=new_features[limb], scales=self.limb_ranges[limb], timestep=self.default_vals['timestep_in_seconds'], feat_show=set_vals['feat_show'])
			limb_act, kin_back = self.run_desired_traj(kin_to_run=attempt_kin, reward_feedback=True, modified_traj=2)

			if self.prev_reward > self.best_reward_so_far:
				self.best_reward_so_far = self.prev_reward
				self.best_features_so_far = copy.deepcopy(new_features)
				#this line has to do with log_directory
				current_logdir = func_base_logdir + '/current_best/'
				#add data to the networks before training
				self.add_training(limb_act, kin_back)
				self.save_features_func(logdir=current_logdir)
				current_logdir = func_base_logdir
			if set_vals['refinement']:
				for limb in self.limb_order:
					self.limbs[limb].train()
			
			print("Best reward so far: ", self.best_reward_so_far)
		input("Learning to move completed, press enter to continue")
		for limb in self.limb_order:
			attempt_kin[limb] = self.feat_to_attempt_single_limb(features=self.best_features_so_far, scales=self.limb_ranges[limb], timestep=self.default_vals['timestep_in_seconds'], feat_show=set_vals['feat_show'])
		self.run_desired_traj(kin_to_run=attempt_kin, reward_feedback=True, modified_traj=2)

		if set_vals['show_final'] == True:
			for limb in self.limb_order:
				self.visualize_results(limb, 1)
		
		print("All rewards: ", self.all_rewards)
		print("Best reward of last run: ", self.prev_reward)
		return self.best_reward_so_far, self.all_rewards
