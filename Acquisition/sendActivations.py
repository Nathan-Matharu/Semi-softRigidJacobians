import zmq
import zwrap
import numpy as np
import math

class zmqWrapperBeginning:
	def __init__(self, pubPort, subIP, setup):
		self.context = zmq.Context()
		self.sub = zwrap.connectsub(self.context, "tcp://"+subIP)
		self.pub = zwrap.bindpub(self.context, "tcp://*:"+pubPort)
		self.setup = setup

	def sendAndReceive(self, activations, stepInMillisec):
		temp = []
		sending = activations
		zwrap.pubmsg(self.pub, self.setup, sending)
		for _ in range(stepInMillisec):
			response = zwrap.submsg(self.sub, self.setup)
		for value in response:
			temp.append(value)
		return temp

def degree2Excurs(degrees, diameter):
	return (math.pi*diameter)*(degrees/360)

pxi1 = "169.254.251.194:5555"
pxi2 = "169.254.22.65:5555"
pubPort = "5557"

motorShaftDiameter = 6 # in mm

sendOff = zmqWrapperBeginning(pubPort, pxi1, zwrap.setups.hand_4_4)

values = [0.0, 0.0, 0.0, 0.0]
outputs = []
slots = [0,0,0,0]
postures = [
	[1.4,0.6,0.6,0.3], # Horizontal--do not touch [1.4,0.6,0.6,0.3]
	[0,0,0,0],
	[0,0,0,0],
	[0.6,0.3,0.5,0.6], #posture 2-- do not touch [0.6,0.3,0.5,0.6]
	[0,0,0,0],
	[0,0,0,0],
	[.15,0.5,0.1,0.76], # Down--do not touch [.15,0.5,0.1,0.8]
	[0,0,0,0],
	[0,0,0,0],
	[0.8,0,0,0],# tendon 1
	[0,0,0,0],
	[0,0,0,0],
	[0,0.8,0,0],# tendon 2
	[0,0,0,0],
	[0,0,0,0],
	[0,0,0.8,0],# tendon 3
	[0,0,0,0],
	[0,0,0,0],
	[0,0,0,0.8], # tendon 4
	[0,0,0,0],
	[0,0,0,0]
]

hardExcursions = [
	[0.8, 0.2, 0.2, 0.2],
	[0,0,0,0],
	[0.2, 1.3, 0.2, 0.2],
	[0,0,0,0],
	[1.1, 0.1, 0.1, 0.1],
	[0,0,0,0]
]

randAct = [[0.8277, 0.615, 0.8826, 0.6306], [0,0,0,0],
[0.7334, 0.2467, 0.9666, 0.6487], [0,0,0,0],
[0.3955, 0.0954, 0.76, 0.7121], [0,0,0,0],
[0.8539, 0.6347, 0.491, 0.8777], [0,0,0,0],
[0.8175, 0.7392, 0.9288, 0.7428], [0,0,0,0],
[0.2737, 0.977, 0.1252, 0.9266], [0,0,0,0],
[0.1474, 0.3422, 0.9354, 0.7624], [0,0,0,0],
[0.0065, 0.2522, 0.3883, 0.3359], [0,0,0,0],
[0.017, 0.9621, 0.5, 0.2475], [0,0,0,0],
[0.2243, 0.5055, 0.619, 0.5238], [0,0,0,0],
[0.2403, 0.1123, 0.6297, 0.5649], [0,0,0,0],
[0.6173, 0.9865, 0.9759, 0.1429], [0,0,0,0],
[0.0289, 0.1396, 0.4391, 0.8531], [0,0,0,0],
[0.3791, 0.906, 0.1893, 0.7517], [0,0,0,0],
[0.5253, 0.7043, 0.3367, 0.9617], [0,0,0,0],
[0.7171, 0.2836, 0.6632, 0.1317], [0,0,0,0],
[0.9611, 0.2714, 0.6713, 0.857], [0,0,0,0],
[0.3474, 0.0631, 0.6995, 0.8838], [0,0,0,0],
[0.7055, 0.1043, 0.6364, 0.9259], [0,0,0,0],
[0.4627, 0.21, 0.1199, 0.1338], [0,0,0,0],
[0.9072, 0.0137, 0.0189, 0.3689], [0,0,0,0],
[0.4702, 0.1559, 0.284, 0.8589], [0,0,0,0],
[0.8178, 0.8512, 0.1949, 0.2333], [0,0,0,0],
[0.0334, 0.6718, 0.5127, 0.4161], [0,0,0,0],
[0.1005, 0.3724, 0.8611, 0.3782], [0,0,0,0],
[0.7318, 0.8033, 0.8943, 0.8519], [0,0,0,0],
[0.214, 0.6838, 0.1748, 0.7541], [0,0,0,0],
[0.2971, 0.4219, 0.0556, 0.2121], [0,0,0,0],
[0.3308, 0.6591, 0.6068, 0.8116], [0,0,0,0],
[0.4502, 0.5392, 0.7866, 0.2765], [0,0,0,0],
[0.7855, 0.3076, 0.0947, 0.591], [0,0,0,0],
[0.2352, 0.1928, 0.5802, 0.3049], [0,0,0,0],
[0.6077, 0.9613, 0.1816, 0.9045], [0,0,0,0],
[0.1065, 0.271, 0.208, 0.2223], [0,0,0,0],
[0.6017, 0.1566, 0.4826, 0.4486], [0,0,0,0],
[0.6017, 0.3939, 0.8049, 0.7936], [0,0,0,0],
[0.7001, 0.2731, 0.182, 0.8762], [0,0,0,0],
[0.2391, 0.1282, 0.2065, 0.4686], [0,0,0,0],
[0.5284, 0.7375, 0.7328, 0.2009], [0,0,0,0],
[0.1618, 0.9262, 0.4433, 0.2655], [0,0,0,0],
[0.9124, 0.6105, 0.5713, 0.7449], [0,0,0,0],
[0.3874, 0.5729, 0.2148, 0.8684], [0,0,0,0],
[0.4255, 0.1277, 0.0492, 0.722], [0,0,0,0],
[0.6976, 0.4582, 0.6165, 0.4312], [0,0,0,0]]

try:
	while True:
		'''
		#for slot in range(len(slots)):
		postureSetExcursions = []
		for i in range(len(postures)):
			#slots[slot] = 0.85
			ham = sendOff.sendAndReceive(postures[i], 1750)
			for p in range(len(ham)):
				ham[p] = round(degree2Excurs(ham[p], motorShaftDiameter), 4)
			postureSetExcursions.append(ham)
			#slots[slot] = 0
		print(str(postureSetExcursions[::2])+"\r")
		'''
		
		postureSetExcursions = []
		for i in range(len(postures)):
			sef = sendOff.sendAndReceive(postures[i], 1750)
			print("Excursions for Posture " + str(i) + " complete")
			input()
			for p in range(len(sef)):
				sef[p] = round(degree2Excurs(sef[p], motorShaftDiameter), 4)
			postureSetExcursions.append(sef)
		print(str(postureSetExcursions)+"\r")
		
		'''
		postureSetExcursions = []
		for i in range(len(randAct)):
			handOff = sendOff.sendAndReceive(randAct[i], 1750)
			for p in range(len(handOff)):
				handOff[p] = round(degree2Excurs(handOff[p], motorShaftDiameter), 4)
			postureSetExcursions.append(handOff)
			print("posture " + str(i) + " complete")
		print(str(postureSetExcursions[::2])+"\r")
		'''
		

except KeyboardInterrupt:
	print("\nCTRL-C Interupt detected")
finally:
	print("\nClosing the python Program")