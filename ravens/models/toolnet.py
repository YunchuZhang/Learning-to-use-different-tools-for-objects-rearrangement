import tensorflow as tf
import numpy as np
from ravens.utils import utils
from ravens.models.resnet import ResNet43_8s, ResNet_tool
from tensorflow_addons import image as tfa_image
import tensorflow_addons as tfa


"""
Tool affordance module
fully convolutional Residual Newwork 43 layers and 8-stride
then with three head to get tool heatmap
"""
	# input_data = tf.keras.layers.Input(shape=input_shape)


class Toolnet:
	"""Attention module."""
	# in_shape 360 720 3
	def __init__(self, in_shape, n_rotations, preprocess, tool_num=3, feat_dim=64, lite=False):

		self.tool_num = tool_num
		self.feat_dim = feat_dim
		self.n_rotations = n_rotations
		self.preprocess = preprocess
		self.in_img = in_shape

		max_dim = np.max(in_shape[:2])

		self.padding = np.zeros((3, 2), dtype=int)
		pad = (max_dim - np.array(in_shape[:2])) / 2
		self.padding[:2] = pad.reshape(2, 1)

		in_shape = np.array(in_shape)
		in_shape += np.sum(self.padding, axis=1)
		in_shape = tuple(in_shape)

		d_in, tool_heats = ResNet_tool(in_shape, 1, self.padding)


		self.model = tf.keras.models.Model(inputs=[d_in], outputs=[tool_heats])
		self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
		self.metric = tf.keras.metrics.Mean(name='loss_toolnet')


	def forward(self, in_img, softmax=True):
		"""Forward pass."""
		in_data = np.pad(in_img, self.padding, mode='constant')
		in_data = self.preprocess(in_data)
		in_shape = (1,) + in_data.shape
		in_data = in_data.reshape(in_shape)
		in_tens = tf.convert_to_tensor(in_data, dtype=tf.float32)

		# Rotate input.
		pivot = np.array(in_data.shape[1:3]) / 2
		rvecs = self.get_se2(self.n_rotations, pivot)
		in_tens = tf.repeat(in_tens, repeats=self.n_rotations, axis=0)
		in_tens = tfa_image.transform(in_tens, rvecs, interpolation='NEAREST')

		# Forward pass.
		in_tens = tf.split(in_tens, self.n_rotations)
		logits=()
		for x in in_tens:
			logits += (self.model(x),)
		logits = tf.concat(logits, axis=0)
		# Rotate back output.
		rvecs = self.get_se2(self.n_rotations, pivot, reverse=True)
		logits = tfa_image.transform(logits, rvecs, interpolation='NEAREST')

		c0 = self.padding[:2, 0]
		c1 = c0 + in_img.shape[:2]
		logits = tf.transpose(logits, [0, 2, 3, 1])
		logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]


		logits = tf.transpose(logits, [3, 1, 2, 0])
		output = tf.reshape(logits, (self.tool_num, np.prod(logits.shape[1:])))
		if softmax:
			output = tf.nn.softmax(output, axis=1)
			output = np.float32(output).reshape((self.tool_num, logits.shape[1], logits.shape[2]))
		return output

	def train(self, in_img, p, theta, tool_id=0, is_mix=False, backprop=True):
		"""Train."""
		tool_coeff = 1.0
		nontool_coeff = 1.0

		self.metric.reset_states()
		with tf.GradientTape() as tape:
			output = self.forward(in_img, softmax=False) # 3, H*W

			# Get single label (for the tool w/ id==tool_id)
			theta_i = theta / (2 * np.pi / self.n_rotations)
			theta_i = np.int32(np.round(theta_i)) % self.n_rotations
			label_size = in_img.shape[:2] + (self.n_rotations,)
			label0 = np.zeros(label_size)
			label0[p[0], p[1], theta_i] = 1
			label0 = label0.reshape(1, np.prod(label0.shape)) # 1, H*W
			label0 = tf.convert_to_tensor(label0, dtype=tf.float32)

			# Get uniform label (for the tools w/ id!=tool_id)
			label1 = np.zeros(label_size)
			label1[:,:,:] = 1.0 / np.prod(label1.shape)
			label1 = label1.reshape(1, np.prod(label1.shape)) # 1, H*W
			label1 = tf.convert_to_tensor(label1, dtype=tf.float32)

			# Get loss
			if is_mix:
				# if a mixture, only supervise the tool w/ id==tool_id
				loss = tf.nn.softmax_cross_entropy_with_logits(label0, output[tool_id:tool_id+1])
				# tool_coeff = 1.0
				# nontool_coeff = 0.05
				# loss = 0.0
				# for tid in range(self.tool_num):
				# 	if tid == tool_id:
				# 		loss += tool_coeff * tf.nn.softmax_cross_entropy_with_logits(label0, output[tid:tid+1])
				# 	else:
				# 		loss += nontool_coeff * tf.nn.softmax_cross_entropy_with_logits(label1, output[tid:tid+1])
			else:
				# if not a mixture, supervise all tools
				tool_coeff = 1.0
				nontool_coeff = 0.5
				loss = 0.0
				for tid in range(self.tool_num):
					if tid == tool_id:
						loss += tool_coeff * tf.nn.softmax_cross_entropy_with_logits(label0, output[tid:tid+1])
					else:
						loss += nontool_coeff * tf.nn.softmax_cross_entropy_with_logits(label1, output[tid:tid+1])
			loss = tf.reduce_mean(loss)

		# Backpropagate
		if backprop:
			grad = tape.gradient(loss, self.model.trainable_variables)
			self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
			self.metric(loss)

		return np.float32(loss), output

	def load(self, path):
		self.model.load_weights(path)

	def save(self, filename):
		self.model.save(filename)

	def get_se2(self, n_rotations, pivot, reverse=False):
		"""Get SE2 rotations discretized into n_rotations angles counter-clockwise."""
		rvecs = []
		for i in range(n_rotations):
			theta = i * 2 * np.pi / n_rotations
			theta = -theta if reverse else theta
			rmat = utils.get_image_transform(theta, (0, 0), pivot)
			rvec = rmat.reshape(-1)[:-1]
			rvecs.append(rvec)
		return np.array(rvecs, dtype=np.float32)