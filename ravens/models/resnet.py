# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resnet module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def identity_block(input_tensor,
									 kernel_size,
									 filters,
									 stage,
									 block,
									 activation=True,
									 include_batchnorm=False):
	"""The identity block is the block that has no conv layer at shortcut.

	Args:
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of
				middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
		activation: If True, include ReLU activation on the output.
		include_batchnorm: If True, include intermediate batchnorm layers.

	Returns:
		Output tensor for the block.
	"""
	filters1, filters2, filters3 = filters
	batchnorm_axis = 3
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = tf.keras.layers.Conv2D(
			filters1, (1, 1),
			dilation_rate=(1, 1),
			kernel_initializer='glorot_uniform',
			name=conv_name_base + '2a')(
					input_tensor)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '2a')(
						x)
	x = tf.keras.layers.ReLU()(x)

	x = tf.keras.layers.Conv2D(
			filters2,
			kernel_size,
			dilation_rate=(1, 1),
			padding='same',
			kernel_initializer='glorot_uniform',
			name=conv_name_base + '2b')(
					x)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '2b')(
						x)
	x = tf.keras.layers.ReLU()(x)

	x = tf.keras.layers.Conv2D(
			filters3, (1, 1),
			dilation_rate=(1, 1),
			kernel_initializer='glorot_uniform',
			name=conv_name_base + '2c')(
					x)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '2c')(
						x)

	x = tf.keras.layers.add([x, input_tensor])

	if activation:
		x = tf.keras.layers.ReLU()(x)
	return x


def conv_block(input_tensor,
							 kernel_size,
							 filters,
							 stage,
							 block,
							 strides=(2, 2),
							 activation=True,
							 include_batchnorm=False):
	"""A block that has a conv layer at shortcut.

	Note that from stage 3,
	the first conv layer at main path is with strides=(2, 2)
	And the shortcut should have strides=(2, 2) as well

	Args:
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of
				middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
		strides: Strides for the first conv layer in the block.
		activation: If True, include ReLU activation on the output.
		include_batchnorm: If True, include intermediate batchnorm layers.

	Returns:
		Output tensor for the block.
	"""
	filters1, filters2, filters3 = filters
	batchnorm_axis = 3
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = tf.keras.layers.Conv2D(
			filters1, (1, 1),
			strides=strides,
			dilation_rate=(1, 1),
			kernel_initializer='glorot_uniform',
			name=conv_name_base + '2a')(
					input_tensor)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '2a')(
						x)
	x = tf.keras.layers.ReLU()(x)

	x = tf.keras.layers.Conv2D(
			filters2,
			kernel_size,
			padding='same',
			dilation_rate=(1, 1),
			kernel_initializer='glorot_uniform',
			name=conv_name_base + '2b')(
					x)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '2b')(
						x)
	x = tf.keras.layers.ReLU()(x)

	x = tf.keras.layers.Conv2D(
			filters3, (1, 1),
			kernel_initializer='glorot_uniform',
			dilation_rate=(1, 1),
			name=conv_name_base + '2c')(
					x)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '2c')(
						x)

	shortcut = tf.keras.layers.Conv2D(
			filters3, (1, 1),
			strides=strides,
			dilation_rate=(1, 1),
			kernel_initializer='glorot_uniform',
			name=conv_name_base + '1')(
					input_tensor)
	if include_batchnorm:
		shortcut = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=bn_name_base + '1')(
						shortcut)

	x = tf.keras.layers.add([x, shortcut])
	if activation:
		x = tf.keras.layers.ReLU()(x)
	return x

def tool_head(input_tensor, feat_dim = 256):
	# input_tensor = tf.transpose(input_tensor,[0, 2, 3, 1])
	input_shape = input_tensor.shape
	x = tf.keras.layers.Conv2D(feat_dim, input_shape = input_shape[2:], data_format='channels_first', kernel_size=3, padding='same',activation='gelu')(input_tensor)
	x = tfa.layers.GroupNormalization(groups=1, axis=2)(x)

	prefix = 'th_'
	x = tf.transpose(tf.reshape(x, [-1,256,90,90]),[0, 2, 3, 1])
	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(
					x)
	x = conv_block(
			x, 3, [128, 128, 128], stage=1, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [128, 128, 128], stage=1, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(
					x)
	x = conv_block(
			x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(
					x) 
	# [None, 720, 720, 64]
	x = conv_block(
			x,
			3, [16, 16, 1],
			stage=9,
			block=prefix + 'a',
			strides=(1, 1),
			activation=False)
	output = identity_block(
			x, 3, [16, 16, 1], stage=3, block=prefix + 'b', activation=False)

	return tf.transpose(tf.reshape(output, [-1,3,720,720,1]),[0, 1, 4, 2, 3])
	'''
	input_shape = x.shape
	x = tf.keras.layers.Conv2D(feat_dim, kernel_size=3, input_shape = input_shape[1:], data_format='channels_first', padding='same',activation='gelu')(x)
	x = tfa.layers.GroupNormalization(groups=1, axis=1)(x)
	x = tf.keras.layers.Conv2D(feat_dim, kernel_size=3, input_shape = input_shape[1:], data_format='channels_first', padding='same',activation='gelu')(x)
	x = tfa.layers.GroupNormalization(groups=1, axis=1)(x)
	output = tf.keras.layers.Conv2D(1, input_shape = input_shape[1:], data_format='channels_first', kernel_size=1, padding='same')(x)
    '''


def integrate(heatmap):
	'''
	Input:
		heatmap: (B, 1, H, W)
	Output:
		grid_coord: (B, 2)
	'''
	B, _, H, W = heatmap.shape
	h = tf.range(H) 
	w = tf.range(W)

	grid_h, grid_w = tf.meshgrid(h,w) # h, w

	# d = tf.constant([B,1,1,1], tf.float32)
	grid_hw = tf.cast(tf.expand_dims(tf.stack([grid_h, grid_w], axis=0),axis=0),tf.float32) # B, 2, H, W
	# grid_hw = tf.tile(grid_hw, d)

	grid_coord = tf.math.reduce_sum(grid_hw * heatmap, axis=[2,3]) # B, 2

	return grid_coord

def ResNet_tool(input_shape,  # pylint: disable=invalid-name
				output_dim,
				padding,
				tool_num = 3,
				feat_dim = 256,
				):
	input_data = tf.keras.layers.Input(shape=input_shape)
	# Initialize fully convolutional Residual Network with 43 layers and
	# 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
	# pick network
	# d_in0, d_out0 = ResNet(input_data, output_dim, prefix='t0_')


	# heatmap = tf.cast(tf.transpose(d_out0, [0, 3, 1, 2]), tf.float64)
	# heatmap = tf.reshape(heatmap, [-1,64,90,90])
	# if softmax:
	# 	output = tf.nn.softmax(output)
	# 	output = np.float32(output).reshape(logits.shape[1:])

	# vis_feature net
	# [None, 720, 720, 64]
	# [None, 90, 90, 256]
	d_in1, d_out1, emb1 = ResNet(input_data, output_dim, prefix='t1_', return_emb=True)
	# import ipdb;ipdb.set_trace()
	tool_emb = tf.Variable(tf.experimental.numpy.random.randn(tool_num, feat_dim), trainable=True)
	tool_filter = tf.Variable(tf.experimental.numpy.random.randn(tool_num, feat_dim, feat_dim), trainable=True)

	# import ipdb;ipdb.set_trace()
	vis_feat = tf.cast(tf.transpose(emb1, [0, 3, 1, 2]),tf.float64) # None C, H, W 256 90 90

	conv2d_layers = []
	
	tool_feats = []
	tool_coords = []
	for tool_id in range(tool_num):
		# tool_feat = tf.identity(vis_feat) * tf.reshape(tool_emb[tool_id],[1,-1,1,1])
		# tool_feat = tf.identity(vis_feat) + tf.reshape(tool_emb[tool_id],[1,-1,1,1])

		tool_feat = tf.nn.conv2d(tf.identity(vis_feat),tf.reshape(tool_filter[tool_id],[1,1,feat_dim,feat_dim]),strides=1, padding="SAME", data_format='NCHW')

		tool_feats.append(tool_feat)

	tool_feats = tf.concat(tool_feats, axis=1)
	tool_feats = tf.reshape(tool_feats, [-1 ,3, 256, 90, 90])

	tool_feats = tool_head(tool_feats, feat_dim)

	h_min, h_max = tf.reduce_min(tool_feats,axis=[2,3,4], keepdims = True), tf.reduce_max(tool_feats,axis=[2,3,4], keepdims = True)
	tool_heats = (tool_feats - h_min)/(h_max - h_min)

	# tool_coord = integrate(tool_heat)
	# tool_coords.append(tool_coord)
        

	#tool_heats = tf.concat(tool_heats, axis=1)
	#tool_coords = tf.stack(tool_coords, axis=1)

	return input_data, tf.squeeze(tool_feats,axis=2)

def ResNet_bc(input_shape,  # pylint: disable=invalid-name
				output_dim,
				padding,
				tool_num = 3,
				feat_dim = 256,
				):
	input_data = tf.keras.layers.Input(shape=input_shape)
	# Initialize fully convolutional Residual Network with 43 layers and
	# 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
	# pick network
	# d_in0, d_out0 = ResNet(input_data, output_dim, prefix='t0_')


	# heatmap = tf.cast(tf.transpose(d_out0, [0, 3, 1, 2]), tf.float64)
	# heatmap = tf.reshape(heatmap, [-1,64,90,90])
	# if softmax:
	# 	output = tf.nn.softmax(output)
	# 	output = np.float32(output).reshape(logits.shape[1:])

	# vis_feature net
	# [None, 720, 720, 64]
	# [None, 90, 90, 256]
	d_in1, d_out1, emb1 = ResNet(input_data, output_dim, prefix='t1_', return_emb=True)

	d_in2, d_out2, emb2 = ResNet(input_data, output_dim, prefix='t2_', return_emb=True)

	x = tf.keras.layers.AveragePooling2D(pool_size=(256,256),strides=(1, 1), padding='valid')(d_out1)
	x = tf.keras.layers.AveragePooling2D(pool_size=(256,256),strides=(1, 1), padding='valid')(x)
	x = tf.keras.layers.AveragePooling2D(pool_size=(128,128),strides=(1, 1), padding='valid')(x)
	x = tf.reduce_mean(x,axis=1)
	x = tf.squeeze(x,axis=2)
	x = tf.keras.layers.Dense(128, activation='relu')(x)
	x = tf.keras.layers.Dense(64, activation='relu')(x)
	x = tf.keras.layers.Dense(6, activation='relu')(x)


	x2 = tf.reduce_mean(d_out2,axis=1)
	x2 = tf.squeeze(x2,axis=2)
	x2 = tf.keras.layers.Dense(128, activation='relu')(x)
	x2 = tf.keras.layers.Dense(64, activation='relu')(x)
	x2 = tf.keras.layers.Dense(3, activation='relu')(x)
	# tool_coord = integrate(tool_heat)
	# tool_coords.append(tool_coord)
        

	#tool_heats = tf.concat(tool_heats, axis=1)
	#tool_coords = tf.stack(tool_coords, axis=1)
	return input_data, tf.concat((x,x2),axis=1)
def ResNet(	input_data,
			output_dim,
			include_batchnorm=False,
			batchnorm_axis=3,
			prefix='',
			cutoff_early=False,
			return_emb = False
								):
	"""Build Resent 43 8s."""

	x = tf.keras.layers.Conv2D(
			64, (3, 3),
			strides=(1, 1),
			padding='same',
			kernel_initializer='glorot_uniform',
			name=prefix + 'conv1')(
					input_data)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=prefix + 'bn_conv1')(
						x)
	x = tf.keras.layers.ReLU()(x)

	if cutoff_early:
		x = conv_block(
				x,
				5, [64, 64, output_dim],
				stage=2,
				block=prefix + 'a',
				strides=(1, 1),
				include_batchnorm=include_batchnorm)
		x = identity_block(
				x,
				5, [64, 64, output_dim],
				stage=2,
				block=prefix + 'b',
				include_batchnorm=include_batchnorm)
		return input_data, x


	x = conv_block(
			x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b')

	x = conv_block(
			x, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'b')

	x = conv_block(
			x, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'b')

	x = conv_block(
			x, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'b')

	x = conv_block(
			x, 3, [256, 256, 256], stage=6, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'b')

	# [None, 90, 90, 256]
	emb = x 

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(
					x)

	x = conv_block(
			x, 3, [128, 128, 128], stage=7, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(
					x)

	x = conv_block(
			x, 3, [64, 64, 64], stage=8, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=8, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(
					x) 
	# [None, 720, 720, 64]
	x = conv_block(
			x,
			3, [16, 16, output_dim],
			stage=9,
			block=prefix + 'a',
			strides=(1, 1),
			activation=False)
	output = identity_block(
			x, 3, [16, 16, output_dim], stage=9, block=prefix + 'b', activation=False)

	if return_emb:
		return input_data, output, emb
	return input_data, output

def ResNet43_8s(input_shape,  # pylint: disable=invalid-name
								output_dim,
								include_batchnorm=False,
								batchnorm_axis=3,
								prefix='',
								cutoff_early=False,
								return_emb = False
								):
	"""Build Resent 43 8s."""
	# TODO(andyzeng): rename to ResNet36_4s

	input_data = tf.keras.layers.Input(shape=input_shape)

	x = tf.keras.layers.Conv2D(
			64, (3, 3),
			strides=(1, 1),
			padding='same',
			kernel_initializer='glorot_uniform',
			name=prefix + 'conv1')(
					input_data)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=prefix + 'bn_conv1')(
						x)
	x = tf.keras.layers.ReLU()(x)

	if cutoff_early:
		x = conv_block(
				x,
				5, [64, 64, output_dim],
				stage=2,
				block=prefix + 'a',
				strides=(1, 1),
				include_batchnorm=include_batchnorm)
		x = identity_block(
				x,
				5, [64, 64, output_dim],
				stage=2,
				block=prefix + 'b',
				include_batchnorm=include_batchnorm)
		return input_data, x

	x = conv_block(
			x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b')

	x = conv_block(
			x, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'b')

	x = conv_block(
			x, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'b')

	x = conv_block(
			x, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'b')

	x = conv_block(
			x, 3, [256, 256, 256], stage=6, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(
					x)

	x = conv_block(
			x, 3, [128, 128, 128], stage=7, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(
					x)

	x = conv_block(
			x, 3, [64, 64, 64], stage=8, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=8, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(
					x)
	emb = x 
	# [None, 720, 720, 64]
	x = conv_block(
			x,
			3, [16, 16, output_dim],
			stage=9,
			block=prefix + 'a',
			strides=(1, 1),
			activation=False)
	output = identity_block(
			x, 3, [16, 16, output_dim], stage=9, block=prefix + 'b', activation=False)

	if return_emb:
		return input_data, output, emb
	return input_data, output


def ResNet36_4s(input_shape,  # pylint: disable=invalid-name
								output_dim,
								include_batchnorm=False,
								batchnorm_axis=3,
								prefix='',
								cutoff_early=False):
	"""Build Resent 36 4s."""
	# TODO(andyzeng): rename to ResNet36_4s

	input_data = tf.keras.layers.Input(shape=input_shape)

	x = tf.keras.layers.Conv2D(
			64, (3, 3),
			strides=(1, 1),
			padding='same',
			kernel_initializer='glorot_uniform',
			name=prefix + 'conv1')(
					input_data)
	if include_batchnorm:
		x = tf.keras.layers.BatchNormalization(
				axis=batchnorm_axis, name=prefix + 'bn_conv1')(
						x)
	x = tf.keras.layers.ReLU()(x)

	if cutoff_early:
		x = conv_block(
				x,
				5, [64, 64, output_dim],
				stage=2,
				block=prefix + 'a',
				strides=(1, 1),
				include_batchnorm=include_batchnorm)
		x = identity_block(
				x,
				5, [64, 64, output_dim],
				stage=2,
				block=prefix + 'b',
				include_batchnorm=include_batchnorm)
		return input_data, x

	x = conv_block(
			x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b')

	x = conv_block(
			x, 3, [64, 64, 64], stage=3, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [64, 64, 64], stage=3, block=prefix + 'b')

	x = conv_block(
			x, 3, [64, 64, 64], stage=4, block=prefix + 'a', strides=(2, 2))
	x = identity_block(x, 3, [64, 64, 64], stage=4, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(
					x)

	x = conv_block(
			x, 3, [64, 64, 64], stage=8, block=prefix + 'a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 64], stage=8, block=prefix + 'b')

	x = tf.keras.layers.UpSampling2D(
			size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(
					x)

	x = conv_block(
			x,
			3, [16, 16, output_dim],
			stage=9,
			block=prefix + 'a',
			strides=(1, 1),
			activation=False)
	output = identity_block(
			x, 3, [16, 16, output_dim], stage=9, block=prefix + 'b', activation=False)

	return input_data, output
