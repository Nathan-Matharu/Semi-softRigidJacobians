import External_Resources.RTBridge as rtb
import numpy as np
#from External_Resources.general2particular import G2P_CORE as core
import matplotlib.pyplot as plt
import math

pxi1 = "169.254.251.194:5555"
pxiWin10 = "169.254.172.223:5555"
pubPort = "5557"
bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, 1500)
#print("test")
def angle2excursion(angles, shaft_diameter_mm):
	temp = []
	for u in angles:
		temp.append((u/360)*np.pi*shaft_diameter_mm) # excursions are in mm?
	return temp

t_sec = 1; # 1 second per movement
dt = 10; # time between send and receives to/from PXI in ms
timesteps = round(t_sec*1000/dt); # how many steps between each position. 1000/10 = 100ms steps
def sigmoid(min=0,max=1):
	inputs = np.linspace(-4.595,4.595,timesteps)
	outputs = round(timesteps)*[min] # stay in position for 100ms to fully bend
	for x in inputs:
		outputs = np.append(outputs,(max-min) / (1 + math.exp(-x)) + min)
	return outputs

def smoothLinActs(activations):
	T0 = []
	T1 = []
	T2 = []
	T3 = []
	for i in range(activations.shape[0]-1):
		T0 = np.append(np.linspace(activations[i, 0],activations[i+1, 0],timesteps))
		T1 = np.append(np.linspace(activations[i, 1],activations[i+1, 1],timesteps))
		T2 = np.append(np.linspace(activations[i, 2],activations[i+1, 2],timesteps))
		T3 = np.append(np.linspace(activations[i, 3],activations[i+1, 3],timesteps))
	smoothActs = np.transpose([T0, T1, T2, T3])

	return smoothActs

def smoothSigActs(activations):
	T0 = []
	T1 = []
	T2 = []
	T3 = []
	for i in range(activations.shape[0]-1):
		T0 = np.append(T0,sigmoid(activations[i, 0],activations[i+1, 0]))
		T1 = np.append(T1,sigmoid(activations[i, 1],activations[i+1, 1]))
		T2 = np.append(T2,sigmoid(activations[i, 2],activations[i+1, 2]))
		T3 = np.append(T3,sigmoid(activations[i, 3],activations[i+1, 3]))
	smoothActs = np.transpose([T0, T1, T2, T3])

	return smoothActs
#t_sec = 15 # default 30
#dt = 1.5

#DAQ samples at 1000 hz, so should have dt of 0.001
#samples = int(t_sec/dt) + 1 # 20 samples (21 - 1) subtracted on line 34
max_in = 0.95	# should be 3.04 volts. maxVolt * max_in == 3.04. maxVolt is on the PXI
min_in = 0.15
min_in2 = 0.15	# minimum input for extensor (tendon 3)
pass_chance = 1	 # Chance that the next activation set is different from the current one


BaselineAct = [0.0, 0.0, 0.0, 0.8] # the "Resting" position. Currently set to straight finger

#   #    #   # Change for Every Run  #   #   #   #   #   #   #   #   #   #   #
bone_length_in_cm = 4.5
run_no = 0 # Start from 0
#   #    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

# Generate Babbling inputs
acts = []
#for _ in range(3):
#	acts.append(core.systemID_input_gen_fcn(samples, pass_chance, max_in, min_in))
#acts.append(core.systemID_input_gen_fcn(samples,pass_chance,max_in,min_in2)) # Add activations for extensor (tendon 3)
#acts = np.transpose(acts)
#acts = np.delete(acts,0,0) # Remove the first row, as its always minumum values

#acts = np.array([[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]])

#t0 vs t1
#acts = np.array([[0.1,0.5,0.1,0.1],[0.1,1,0.1,0.1],[0.5,0.1,0.1,0.1],[0.5,0.5,0.1,0.1],[0.5,1,0.1,0.1],[1,0.1,0.1,0.1],[1,0.5,0.1,0.1],[1,1,0.1,0.1]])

#t0 vs t2
#acts = np.array([[0.1,0.1,0.5,0.1],[0.1,0.1,1,0.1],[0.5,0.1,0.1,0.1],[0.5,0.1,0.5,0.1],[0.5,0.1,1,0.1],[1,0.1,0.1,0.1],[1,0.1,0.5,0.1],[1,0.1,1,0.1]])

#t0 vs t3
#acts = np.array([[0.1,0.1,0.1,0.5],[0.1,0.1,0.1,1],[0.5,0.1,0.1,0.1],[0.5,0.1,0.1,0.5],[0.5,0.1,0.1,1],[1,0.1,0.1,0.1],[1,0.1,0.1,0.5],[1,0.1,0.1,1]])

# t1 vs t2
#acts = np.array([[0.1,0.1,0.5,0.1],[0.1,0.1,1,0.1],[0.1,0.5,0.1,0.1],[0.1,0.5,0.5,0.1],[0.1,0.5,1,0.1],[0.1,1,0.1,0.1],[0.1,1,0.5,0.1],[0.1,1,1,0.1]])

# t1 vs t3
#acts = np.array([[0.1,0.1,0.1,0.5],[0.1,0.1,0.1,1],[0.1,0.5,0.1,0.1],[0.1,0.5,0.1,0.5],[0.1,0.5,0.1,1],[0.1,1,0.1,0.1],[0.1,1,0.1,0.5],[0.1,1,0.1,1]])

# t2 vs t3
#acts = np.array([[0.1,0.1,0.1,0.5],[0.1,0.1,0.1,1],[0.1,0.1,0.5,0.1],[0.1,0.1,0.5,0.5],[0.1,0.1,0.5,1],[0.1,0.1,1,0.1],[0.1,0.1,1,0.5],[0.1,0.1,1,1]])

# vary each finger
mT = 0.1 # muscleTone
acts = np.array([[mT, mT, mT, mT],[mT, mT, mT, 1],[mT, mT, 1, mT],[mT, 1, mT, mT],[1, mT, mT, mT],[mT, mT, 1, 1],[mT, 1, mT, 1],[1, mT, mT, 1],[1, 1, mT, mT],[1, mT, 1, mT],[mT, 1, 1, mT],[mT, 1, 1, 1],[1, mT, 1, 1],[1, 1, mT, 1],[1, 1, 1, mT],[1, 1, 1, 1]])

# add baseline position between each activation
# Create an array of zeros with an additional row for insertion
new_array = np.zeros((acts.shape[0] * 2 - 1, acts.shape[1]))
for i in range(acts.shape[0]):
    new_array[i * 2,:] = acts[i,:]
    if i != acts.shape[0] - 1:
        new_array[i * 2 + 1] = BaselineAct

acts = new_array


# manually add specific position
#acts = np.delete(acts,-1,axis=0)
#acts = np.vstack((acts,[0, 0, 0, 0])) 

#print(acts) #activations range from min_in to max_in and have [pass_chance] chance to change every set
sActs = smoothSigActs(acts)
sActs = np.vstack((sActs,round(5*timesteps)*[acts[-1,:]])) # stay at end position for 500 ms
#print(sActs)

#file = open('./Scratch.csv', 'w+')
#file.write('A0,A1,A2,A3,\n')
#for i in range(len(sActs)):
#	for act in sActs[i]:
#		file.write(str(act)+',')
#	file.write('\n')
#file.close()
#"""
# Connect to PXI, then send activations and receive back encoder values
bridge.startConnection()
excursions = []
returned_excursions = []
#starting_excursions = bridge.sendAndReceive([0.2]*4, 2000) # Resting position. I want to see how much using these values vs values in for loop makes a difference
starting_excursions = bridge.sendAndReceive(BaselineAct, 2000)
print("starting excursions now:")
for activation_set in sActs:
	#print("activation")
	returned_excursions = bridge.sendAndReceive(activation_set,10) # Do the activation set (row in acts). second term is time before next activation in milliseconds
	excursions.append(returned_excursions.copy()) # record excursions
	#_ = bridge.sendAndReceive(BaselineAct, 1000) # Resting position	---- sendandReceive(self, activations, stepInMillisec=None, npArray=False) self is assumed(?), activations are in volts?
	#starting_excursions.append(_.copy()) # Record resting position excursion


# Subtract starting values to get true values
_ = bridge.sendAndReceive([0.05]*4, 2) # this is the relaxed position at the end of a trial set
for u in excursions:
	for v in range(len(u)):
		u[v] -= starting_excursions[v]

print(excursions)

# Convert encoder values to tendon excursions
tendons = []
for j in range(4):
	tendons.append(angle2excursion([i[j] for i in excursions], 6))

#Save excursions to CSV File
tendons_for_file = np.transpose(tendons)
file = open('./CSV_files/'+str(bone_length_in_cm)+'cmBones_run_no_'+str(run_no)+'.csv', 'w+')
file.write('A0;A1;A2;A3;T0;T1;T2;T3;\n')
for i in range(len(sActs)):
	for act in sActs[i]:
		file.write(str(act)+';')
	for excur in tendons_for_file[i]:
		file.write(str(excur)+';')
	file.write('\n')
file.close()


#"""
"""
steps = np.linspace(1, 2000, 2000)

f, ax = plt.subplots(2, 2)
ax[0][0].plot(steps, tendons[0])

ax[0][1].plot(steps, tendons[1])

ax[1][0].plot(steps, tendons[2])

ax[1][1].plot(steps, tendons[3])

ax[0][0].set_title("Tendon 1") #TODO: give these the names of each tendon anatomically
ax[0][1].set_title("Tendon 2")
ax[1][0].set_title("Tendon 3")
ax[1][1].set_title("Tendon 4")

plt.show()

print("\n\nTest has completed")
"""


#input("Program has completed. Press [ENTER] to continue");
