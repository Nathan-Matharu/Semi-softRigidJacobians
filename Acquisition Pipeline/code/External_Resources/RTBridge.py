import sys
import zmq
import struct
from enum import Enum
import numpy as np

class setups(Enum):
	cat_3_2 = [3,2]
	cat_12_8 = [12,8]
	hand_3_3 = [3,3]
	hand_4_4 = [4,4]
	hand_12_12 = [12,12]

# Connects subscriber socket
def rtb_connectsub(context, subaddr) :
	socket = context.socket(zmq.SUB)
	socket.connect(subaddr)
	zip_filter = ""
	socket.setsockopt_string(zmq.SUBSCRIBE, zip_filter)
	print("Python Subscriber connection successful!")
	return socket

# Binds publisher socket
def rtb_initpub(context, pubaddr) :
	socket = context.socket(zmq.PUB)
	socket.bind(pubaddr)
	print("Python Publisher connection successful!")
	return socket

# Publishes a message
def rtb_publishMsg(pubsocket, setup_type, values) :
	#twelve value max currently
	size = struct.calcsize('ffffffffffff')

	#packedmsg = struct.pack('ffffffffffff', float1, float2, float3)

	if(setup_type == setups.cat_3_2):
		if len(values) > 2:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],values[2],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
		elif len(values) == 2:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
		else:
			raise IndexError("Not enough input values")
	elif(setup_type == setups.cat_12_8):
		if len(values) > 8:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11])
		elif len(values) == 8:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],0.0,0.0,0.0,0.0)
		else:
			raise IndexError("Not enough input values")
	elif(setup_type == setups.hand_3_3):
		if len(values) >=3:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],values[2],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
		else:
			raise IndexError("Not enough input values")
	elif(setup_type == setups.hand_4_4):
		if len(values) >= 4:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],values[2],values[3],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
		else:
			raise IndexError("Not enough input values")
	elif(setup_type == setups.hand_12_12):
		if len(values) >= 12:
			packedmsg = struct.pack('ffffffffffff',
			values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11])
		else:
			raise IndexError("Not enough input values")
	else:
		raise ValueError("Incompatible setup type")

	pubsocket.send(packedmsg)

# Subscribes, and prints if it receives a message
def rtb_receiveMsg(subsocket, setup_type) :
	msg = subsocket.recv()
	#twelve float values because that is the current max
	recv_tuple = struct.unpack('ffffffffffff', msg)

	outs = ()

	if(setup_type == setups.cat_3_2):
		if recv_tuple[2] != 0:
			outs = recv_tuple[:3]
		else:
			outs = recv_tuple[:2]
	elif(setup_type == setups.cat_12_8):
		if recv_tuple[8] != 0:
			outs = recv_tuple[:12]
		else:
			outs = recv_tuple[:8]
	elif(setup_type == setups.hand_3_3):
		outs = recv_tuple[:3]
	elif(setup_type == setups.hand_4_4):
		outs = recv_tuple[:4]
	elif(setup_type == setups.hand_12_12):
		outs = recv_tuple
	else:
		raise ValueError("Incompatible setup type")

	return outs
	sys.stdout.flush()

class BridgeSetup:
	def __init__(self, pubPort, subIP, setup, milliTimeStep=20):
		self.context = zmq.Context()
		self.sub = rtb_connectsub(self.context, "tcp://"+subIP)
		self.pub = rtb_initpub(self.context, "tcp://*:"+pubPort)
		self.setup = setup
		self.timeStep = milliTimeStep

	def sendAndReceive(self, activations, stepInMillisec=None, npArray=False):
		if stepInMillisec == None:
			stepInMillisec = self.timeStep

		temp = []

		rtb_publishMsg(self.pub, self.setup, activations)
		for _ in range(stepInMillisec):
			response = rtb_receiveMsg(self.sub, self.setup)
		for value in response:
			temp.append(value)

		if npArray:
			return np.array(temp)
		else:
			return temp

	def startConnection(self):
		#this is designed to prime the system so that any startup delay occurs here and now when real data is being transferred
		throwaway = []
		sendout = self.setup.value[0]
		for _ in range(sendout):
			throwaway.append(0.05)
		for _ in range(200):
			self.sendAndReceive(throwaway, stepInMillisec=50)
		print("completed Startup")
