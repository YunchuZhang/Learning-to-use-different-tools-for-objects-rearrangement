import tensorflow as tf
import numpy as np
from ravens.utils import utils
from ravens.models.resnet import ResNet43_8s, ResNet_tool, ResNet_bc
from tensorflow_addons import image as tfa_image
import tensorflow_addons as tfa


"""
Tool affordance module
fully convolutional Residual Newwork 43 layers and 8-stride
then with three head to get tool heatmap
"""
	# input_data = tf.keras.layers.Input(shape=input_shape)


class BCnet:
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

		d_in, pred = ResNet_bc(in_shape, 1, self.padding)

		self.model = tf.keras.models.Model(inputs=[d_in], outputs=[pred])
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
		# rvecs = self.get_se2(self.n_rotations, pivot, reverse=True)
		# logits = tfa_image.transform(logits, rvecs, interpolation='NEAREST')

		# c0 = self.padding[:2, 0]
		# c1 = c0 + in_img.shape[:2]
		# logits = tf.transpose(logits, [0, 2, 3, 1])
		# logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]


		# logits = tf.transpose(logits, [3, 1, 2, 0])
		# output = tf.reshape(logits, (self.tool_num, np.prod(logits.shape[1:])))
		# if softmax:
		# 	output = tf.nn.softmax(output, axis=1)
		# 	output = np.float32(output).reshape((self.tool_num, logits.shape[1], logits.shape[2]))
		return logits

	def train(self, in_img, p, theta, p1, p1_theta,tool_id=0, is_mix=False, backprop=True):
		"""Train."""
		tool_coeff = 1.0
		nontool_coeff = 1.0

		self.metric.reset_states()
		with tf.GradientTape() as tape:
			output = self.forward(in_img, softmax=False) # B,7 B 9
			label = np.zeros([1,3])
			label[0][tool_id] = 1
			label = tf.convert_to_tensor(label, dtype=tf.int32)

			# loss = nontool_coeff * tf.nn.l2_loss(output[:,:2] - p)
			# loss += nontool_coeff * tf.nn.l2_loss(output[:,2:4] - p1)
			# loss += tf.nn.softmax_cross_entropy_with_logits(label, output[:,4:7])


			loss = nontool_coeff * tf.nn.l2_loss(output[:,:3] - p)
			loss += nontool_coeff * tf.nn.l2_loss(output[:,3:6] - p1)
			loss += tf.nn.softmax_cross_entropy_with_logits(label, output[:,6:9])


			loss = tf.reduce_mean(loss)

		# Backpropagate
		if backprop:
			grad = tape.gradient(loss, self.model.trainable_variables)
			self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
			self.metric(loss)

		return np.float32(loss), loss

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