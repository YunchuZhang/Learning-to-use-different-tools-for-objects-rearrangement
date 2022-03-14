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

"""Transporter Agent."""

import os

import numpy as np
from ravens.models.attention import Attention
from ravens.models.transport import Transport
from ravens.models.toolnet import Toolnet
from ravens.models.bcnet import BCnet
from ravens.models.transport_ablation import TransportPerPixelLoss
from ravens.models.transport_goal import TransportGoal
from ravens.tasks import cameras
from ravens.utils import utils
from PIL import Image
from ravens.utils.color_jitter import ColorJitter,adjust_hue
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import cv2

 
def transfer(arg):
		return tf.convert_to_tensor(arg)
class TransporterAgent:
	"""Agent that uses Transporter Networks."""

	def __init__(self, name, task, root_dir, n_rotations=36):
		self.name = name
		self.task = task
		self.total_steps = 0
		self.crop_size = 64
		self.n_rotations = n_rotations
		self.pix_size = 0.003125
		self.pix_size = 0.005
		self.pix_size = 0.0015625
		self.pix_size = 0.0015
		self.in_shape = (320, 160, 6)
		self.in_shape = (640, 480, 6)
		self.in_shape = (360, 720, 6)
		self.cam_config = cameras.RealSenseD415.CONFIG
		self.cam_config = cameras.Real.CONFIG
		self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
		self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
		self.bounds = np.array([[-0.35, 0.45], [-1., 0.6], [0.4, 1]])
		self.bounds = np.array([[-0.33, 0.42], [-0.5, 0.5], [0.4, 1]])
		self.bounds = np.array([[-0.58, 0.5], [-0.325, 0.215], [0.4, 1]])
	def get_image(self, obs, jitter = False):
		"""Stack color and height images image."""

		# if self.use_goal_image:
		#   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
		#   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
		#   input_image = np.concatenate((input_image, goal_image), axis=2)
		#   assert input_image.shape[2] == 12, input_image.shape

		# Get color and height maps from RGB-D images.
		cmap, hmap = utils.get_fused_heightmap(
				obs, self.cam_config, self.bounds, self.pix_size)
		img = np.concatenate((cmap,
													hmap[Ellipsis, None],
													hmap[Ellipsis, None],
													hmap[Ellipsis, None]), axis=2)
		assert img.shape == self.in_shape, img.shape


		_transform_dict = {'brightness':0.2, 'contrast':0.2, 'sharpness':0.2, 'color':0.2}
		_color_jitter = ColorJitter(_transform_dict)


		if jitter:
			# import ipdb;ipdb.set_trace()
			img_ = Image.fromarray(np.uint8(img[:,:,:3]))
			img_ = _color_jitter(img_)
			hue_factor = np.random.uniform(-0.5,0.5)
			img_ = adjust_hue(img_,hue_factor)
			img_ = np.array(img_) 
			# import ipdb;ipdb.set_trace()
			return_img = np.concatenate((img_,
														hmap[Ellipsis, None],
														hmap[Ellipsis, None],
														hmap[Ellipsis, None]), axis=2)
			return return_img
		return img

	def get_sample(self, dataset, augment=True):
		"""Get a dataset sample.

		Args:
			dataset: a ravens.Dataset (train or validation)
			augment: if True, perform data augmentation.

		Returns:
			tuple of data for training:
				(input_image, p0, p0_theta, p1, p1_theta)
			tuple additionally includes (z, roll, pitch) if self.six_dof
			if self.use_goal_image, then the goal image is stacked with the
			current image in `input_image`. If splitting up current and goal
			images is desired, it should be done outside this method.
		"""

		(obs, act, _, _), _ = dataset.sample()
		img = self.get_image(obs)

		# Get training labels from data sample.
		p0_xyz, p0_xyzw = act['pose0']
		p1_xyz, p1_xyzw = act['pose1']
		# import ipdb;ipdb.set_trace()
		p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
		# use positive??
		p0_theta = np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
		p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
		p1_theta = np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
		p1_theta = p1_theta - p0_theta
		p0_theta = 0
		# Data augmentation.
		if augment:
			img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

		return img, p0, p0_theta, p1, p1_theta
	def plot_to_tensor(self,feature,figsize=(3.2,1.6)):
		draw_feat = feature.numpy()
		cmap = plt.get_cmap('inferno')
		figure = plt.figure(figsize=figsize)
		plt.imshow(draw_feat[0],cmap = cmap)
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		plt.close(figure)
		buf.seek(0)

		image = tf.image.decode_png(buf.getvalue(), channels=4)
		# Add the batch dimension
		image = tf.expand_dims(image, 0)
		return image
	def train(self, dataset, writer=None):
		"""Train on a dataset sample for 1 iteration.

		Args:
			dataset: a ravens.Dataset.
			writer: a TF summary writer (for tensorboard).
		"""
		h,w,c = self.in_shape
		tf.keras.backend.set_learning_phase(1)
		img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset)
		# import ipdb;ipdb.set_trace()
		# Get training losses.
		step = self.total_steps + 1
		loss0,feature0 = self.attention.train(img, p0, p0_theta)
		if isinstance(self.transport, Attention):
			loss1 = self.transport.train(img, p1, p1_theta)
		else:
			loss1, feature1 = self.transport.train(img, p0, p1, p1_theta)
		with writer.as_default():
			sc = tf.summary.scalar
			rgb = tf.reshape(img[:,:,:3],[1, h, w,3])
			rgb = tf.cast(rgb, dtype=tf.uint8)
			depth = tf.reshape(img[:,:,3],[1, h, w, 1])
			depth = (depth - tf.reduce_min(depth))/(tf.reduce_max(depth) - tf.reduce_min(depth))

			feature0 = (feature0 - tf.reduce_min(feature0))/(tf.reduce_max(feature0) - tf.reduce_min(feature0))
			pick_feat = tf.reshape(feature0,[1, h, w,1])
			# tf.unravel_index(indices, dims, name=None)
			angle = tf.math.argmax(tf.reshape(feature1,(h*w,36)),axis=0)
			place_feat = tf.reshape(feature1[:,:,:,tf.math.argmax(angle)],[1,h,w,1])
			place_feat = (place_feat - tf.reduce_min(place_feat))/(tf.reduce_max(place_feat) - tf.reduce_min(place_feat))

			if step %100 == 0:
				tf.summary.image("rgb", rgb, step=step)
				tf.summary.image("depth", depth, step=step)
				# tf.summary.image("pick_feat_color", self.plot_to_tensor(pick_feat,figsize=(12.8,6.4)), step=step)
				# tf.summary.image("place_feat_color", self.plot_to_tensor(place_feat,figsize=(12.8,6.4)), step=step)     
				tf.summary.image("pick_feat", pick_feat, step=step)
				tf.summary.image("place_feat",place_feat, step=step)

			# rgb_pick = rgb 
			# rgb_place = rgb 

			# f0_min = tf.reduce_min(pick_feat)
			# f0_max = tf.reduce_max(pick_feat)

			# f1_min = tf.reduce_min(place_feat)
			# f1_max = tf.reduce_max(place_feat)


			# new_tensor = tf.Variable(rgb_pick)
			# new_tensor[:,:,:,1].assign(rgb_pick[:,:,:,1]+tf.squeeze(tf.cast((pick_feat-f0_min)/(f0_max-f0_min)*255,dtype=tf.uint8),-1))
			# rgb_pick = transfer(new_tensor)

			# new_tensor = tf.Variable(rgb_place)
			# new_tensor[:,:,:,1].assign(rgb_place[:,:,:,1]+tf.squeeze(tf.cast((place_feat-f1_min)/(f1_max-f1_min)*255,dtype=tf.uint8),-1))
			# rgb_place = transfer(new_tensor)

			# rgb_pick[:,:,:,1] += tf.squeeze(tf.cast((pick_feat-f0_min)/(f0_max-f0_min)*255,dtype=tf.uint8),-1)
			# rgb_place[:,:,:,1] += tf.squeeze(tf,cast((place_feat-f1_min)/(f1_max-f1_min)*255,dtype=tf.uint8),-1)
			# tf.summary.image("rgb_pick", rgb_pick, step=step)
			# tf.summary.image("rgb_place", rgb_place, step=step)

			# tf.summary.image("norm_pick", (pick_feat-f0_min)/(f0_max-f0_min), step=step)
			# tf.summary.image("norm_place", (place_feat-f1_min)/(f1_max-f1_min), step=step)
				sc('train_loss/attention', loss0, step)
				sc('train_loss/transport', loss1, step)
		print(f'Train Iter: {step} Loss: {loss0:.4f} {loss1:.4f}')
		self.total_steps = step


	def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
		"""Test on a validation dataset for 10 iterations."""
		print('Skipping validation.')

	def test(self, obs, cur=0, info=None, goal=None, vis=False, p = None):  # pylint: disable=unused-argument
		"""Run inference and return best action given visual observations."""
		tf.keras.backend.set_learning_phase(0)
		# Get heightmap from RGB-D images.
		img = self.get_image(obs)
		rgb = img[:,:,:3][:,:,::-1]
		rgb = np.array(rgb, dtype=np.uint8)
		depth = img[:,:,3]
		depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
		depth = np.uint8(depth*255)
		# Attention model forward pass.
		pick_conf = self.attention.forward(img)
		argmax = np.argmax(pick_conf)
		argmax = np.unravel_index(argmax, shape=pick_conf.shape)
		p0_pix = argmax[:2]
		p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
		print("predict p0_theta:{}",p0_theta)

		if p != None:
			p0_pix = p
		# Transport model forward pass.
		place_conf = self.transport.forward(img, p0_pix)
		argmax = np.argmax(place_conf)
		argmax = np.unravel_index(argmax, shape=place_conf.shape)
		p1_pix = argmax[:2]
		p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
		print("predict p1_theta:{}",p1_theta*180/np.pi)

		if vis:
			pick_conf = (pick_conf - pick_conf.min())/(pick_conf.max()-pick_conf.min())
			pick_conf = np.uint8(pick_conf*255)

			place_conf = place_conf[:,:,argmax[2]]
			place_conf = (place_conf - place_conf.min())/(place_conf.max()-place_conf.min())
			place_conf = np.uint8(place_conf*255)

			rgb = cv2.circle(rgb, p0_pix[::-1], 4, (0,0,255), 1)
			# rgb = cv2.circle(rgb, [230,265], 4, (0,0,255), 1)
			rgb = cv2.circle(rgb, p1_pix[::-1], 4, (0,255,0), 1)

			angle = str(int(p1_theta*180/np.pi))
			cv2.imwrite("logs/pick_conf{}.png".format(cur),pick_conf)
			cv2.imwrite("logs/place_conf{}_{}.png".format(cur,angle),place_conf)
			cv2.imwrite("logs/rgb{}.png".format(cur),rgb)
			cv2.imwrite("logs/depth{}.png".format(cur),depth)
		# Pixels to end effector poses.
		hmap = img[:, :, 3]
		p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
		p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
		p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, p0_theta))
		p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, p1_theta))
		return {
				'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
				'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
		}

	def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
		"""Run inference and return best action given visual observations."""
		tf.keras.backend.set_learning_phase(0)

		# Get heightmap from RGB-D images.
		img = self.get_image(obs)

		# Attention model forward pass.
		pick_conf = self.attention.forward(img)
		argmax = np.argmax(pick_conf)
		argmax = np.unravel_index(argmax, shape=pick_conf.shape)
		p0_pix = argmax[:2]
		p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

		# Transport model forward pass.
		place_conf = self.transport.forward(img, p0_pix)
		argmax = np.argmax(place_conf)
		argmax = np.unravel_index(argmax, shape=place_conf.shape)
		p1_pix = argmax[:2]
		p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

		# Pixels to end effector poses.
		hmap = img[:, :, 3]
		p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
		p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
		p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
		p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
		return {
				'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
				'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
		}

		# TODO(andyzeng) cleanup goal-conditioned model.

		# Make a goal image if needed, and for consistency stack with input.
		# if self.use_goal_image:
		#   cmap_g, hmap_g = utils.get_fused_heightmap(goal, self.cam_config)
		#   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
		#   input_image = np.concatenate((input_image, goal_image), axis=2)
		#   assert input_image.shape[2] == 12, input_image.shape

		# if self.use_goal_image:
		#   half = int(input_image.shape[2] / 2)
		#   input_only = input_image[:, :, :half]  # ignore goal portion
		#   pick_conf = self.attention.forward(input_only)
		# else:
		# if isinstance(self.transport, TransportGoal):
		#   half = int(input_image.shape[2] / 2)
		#   img_curr = input_image[:, :, :half]
		#   img_goal = input_image[:, :, half:]
		#   place_conf = self.transport.forward(img_curr, img_goal, p0_pix)

	def load(self, n_iter):
		"""Load pre-trained models."""
		print(f'Loading pre-trained model at {n_iter} iterations.')
		attention_fname = 'attention-ckpt-%d.h5' % n_iter
		transport_fname = 'transport-ckpt-%d.h5' % n_iter
		attention_fname = os.path.join(self.models_dir, attention_fname)
		transport_fname = os.path.join(self.models_dir, transport_fname)
		self.attention.load(attention_fname)
		self.transport.load(transport_fname)
		self.total_steps = n_iter

	def save(self):
		"""Save models."""
		if not tf.io.gfile.exists(self.models_dir):
			tf.io.gfile.makedirs(self.models_dir)
		attention_fname = 'attention-ckpt-%d.h5' % self.total_steps
		transport_fname = 'transport-ckpt-%d.h5' % self.total_steps
		attention_fname = os.path.join(self.models_dir, attention_fname)
		transport_fname = os.path.join(self.models_dir, transport_fname)
		self.attention.save(attention_fname)
		self.transport.save(transport_fname)

#-----------------------------------------------------------------------------
# Other Transporter Variants
#-----------------------------------------------------------------------------
class ToolTransporterAgent(TransporterAgent):

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)
		self.tool_num = 3
		self.attention = Toolnet(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess,
				tool_num=self.tool_num)
		self.transport = Transport(
				in_shape=self.in_shape,
				n_rotations=self.n_rotations,
				crop_size=self.crop_size,
				preprocess=utils.preprocess)

	def get_image(self, obs):
		"""Stack color and height images image."""

		# if self.use_goal_image:
		#   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
		#   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
		#   input_image = np.concatenate((input_image, goal_image), axis=2)
		#   assert input_image.shape[2] == 12, input_image.shape

		# Get color and height maps from RGB-D images.
		cmap, hmap = utils.get_fused_heightmap(
				obs, self.cam_config, self.bounds, self.pix_size)
		img = np.concatenate((cmap,
													hmap[Ellipsis, None],
													hmap[Ellipsis, None],
													hmap[Ellipsis, None]), axis=2)
		assert img.shape == self.in_shape, img.shape
		return img

	def get_sample(self, dataset, augment=True):
		"""Get a dataset sample.

		Args:
			dataset: a ravens.Dataset (train or validation)
			augment: if True, perform data augmentation.

		Returns:
			tuple of data for training:
				(input_image, p0, p0_theta, p1, p1_theta)
			tuple additionally includes (z, roll, pitch) if self.six_dof
			if self.use_goal_image, then the goal image is stacked with the
			current image in `input_image`. If splitting up current and goal
			images is desired, it should be done outside this method.
		"""

		(obs, act, _, _), _ = dataset.sample()
		img = self.get_image(obs)

		# Get training labels from data sample.
		p0_xyz, p0_xyzw = act['pose0']
		p1_xyz, p1_xyzw = act['pose1']
		tool_id = act['tid']
		is_mix = act['is_mix']
		# import ipdb;ipdb.set_trace()
		p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
		# use positive??
		p0_theta = np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
		p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
		p1_theta = np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
		p1_theta = p1_theta - p0_theta
		p0_theta = 0
		# Data augmentation.
		if augment:
			img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

		return img, p0, p0_theta, p1, p1_theta, tool_id, is_mix
	def plot_to_tensor(self,feature,figsize=(3.2,1.6)):
		draw_feat = feature.numpy()
		cmap = plt.get_cmap('inferno')
		figure = plt.figure(figsize=figsize)
		plt.imshow(draw_feat[0],cmap = cmap)
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		plt.close(figure)
		buf.seek(0)

		image = tf.image.decode_png(buf.getvalue(), channels=4)
		# Add the batch dimension
		image = tf.expand_dims(image, 0)
		return image
	def train(self, dataset, writer=None):
		"""Train on a dataset sample for 1 iteration.

		Args:
			dataset: a ravens.Dataset.
			writer: a TF summary writer (for tensorboard).
		"""
		h,w,c = self.in_shape
		tf.keras.backend.set_learning_phase(1)
		img, p0, p0_theta, p1, p1_theta, tool_id, is_mix = self.get_sample(dataset)
		# import ipdb;ipdb.set_trace()
		# Get training losses.
		step = self.total_steps + 1
		loss0,feature0 = self.attention.train(img, p0, p0_theta, tool_id = tool_id, is_mix = is_mix)
		if isinstance(self.transport, Attention):
			loss1 = self.transport.train(img, p1, p1_theta)
		else:
			loss1, feature1 = self.transport.train(img, p0, p1, p1_theta)
		
		with writer.as_default():
			sc = tf.summary.scalar
			rgb = tf.reshape(img[:,:,:3],[1, h, w,3])
			rgb = tf.cast(rgb, dtype=tf.uint8)
			depth = tf.reshape(img[:,:,3],[1, h, w, 1])
			depth = (depth - tf.reduce_min(depth))/(tf.reduce_max(depth) - tf.reduce_min(depth))

			# visualize pick feat
			assert (self.tool_num == feature0.shape[0])
			pick_feats = []
			feature0 = (feature0 - tf.reduce_min(feature0))/(tf.reduce_max(feature0) - tf.reduce_min(feature0))
			for tid in range(self.tool_num):
				feat = feature0[tid:tid+1]

				pick_feat = tf.reshape(feat,[1, h, w,1])
				pick_feats.append(pick_feat)

			# visualize place feat
			angle = tf.math.argmax(tf.reshape(feature1,(h*w,36)),axis=0)
			place_feat = tf.reshape(feature1[:,:,:,tf.math.argmax(angle)],[1,h,w,1])
			place_feat = (place_feat - tf.reduce_min(place_feat))/(tf.reduce_max(place_feat) - tf.reduce_min(place_feat))

			if step %100 == 0:
				tf.summary.image("rgb", rgb, step=step)
				tf.summary.image("depth", depth, step=step)
				for tid in range(self.tool_num):
					tf.summary.image("pick_feat_tool{0}".format(tid), pick_feats[tid], step=step)
				tf.summary.image("place_feat",place_feat, step=step)

				sc('train_loss/attention', loss0, step)
				sc('train_loss/transport', loss1, step)
		print(f'Train Iter: {step} Loss: {loss0:.4f} {loss1:.4f}')
		self.total_steps = step



	def test(self, obs, cur=0, info=None, goal=None, vis=False):  # pylint: disable=unused-argument
		"""Run inference and return best action given visual observations."""
		tf.keras.backend.set_learning_phase(0)
		# Get heightmap from RGB-D images.
		img = self.get_image(obs)
		rgb = img[:,:,:3][:,:,::-1]
		rgb = np.array(rgb, dtype=np.uint8)
		depth = img[:,:,3]
		depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
		depth = np.uint8(depth*255)
		
		# Attention model forward pass.
		pick_conf = self.attention.forward(img) # original: 360,720,1. ours: 3,360,720
		# import ipdb;ipdb.set_trace()
		do_integrate = False
		if do_integrate:
			pass 
		else:
			# argmax = np.argmax(pick_conf[:,:,:400])
			argmax = np.argmax(pick_conf)
			argmax = np.unravel_index(argmax, shape=pick_conf.shape)
			# argmax = np.unravel_index(argmax, shape=(3,360,400))
			tool_id = argmax[0]
			p0_pix = argmax[1:]
			p0_theta = 0.0
		print("predict p0_theta:{0}, picked tool:{1}".format(p0_theta, tool_id))

		# if p != None:
		# 	p0_pix = p
		# Transport model forward pass.
		place_conf = self.transport.forward(img, p0_pix)
		argmax = np.argmax(place_conf)
		argmax = np.unravel_index(argmax, shape=place_conf.shape)
		p1_pix = argmax[:2]
		p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
		print("predict p1_theta:{}",p1_theta*180/np.pi)

		if vis:
			pick_confs = []
			pick_conf = (pick_conf - pick_conf.min())/(pick_conf.max()-pick_conf.min()) # single max out of 3 tools
			for tid in range(self.tool_num):
				#pick_conf_i = (pick_conf[tid] - pick_conf[tid].min())/(pick_conf[tid].max()-pick_conf[tid].min())
				pick_conf_i = np.uint8(pick_conf[tid]*255)
				pick_conf_i = np.expand_dims(pick_conf_i, axis=-1)
				pick_confs.append(pick_conf_i)

			place_conf = place_conf[:,:,argmax[2]]
			place_conf = (place_conf - place_conf.min())/(place_conf.max()-place_conf.min())
			place_conf = np.uint8(place_conf*255)

			rgb = cv2.circle(rgb, p0_pix[::-1], 4, (0,0,255), 1)
			rgb = cv2.circle(rgb, p1_pix[::-1], 4, (0,255,0), 1)

			angle = str(int(p1_theta*180/np.pi))
			for tid in range(self.tool_num):
				cv2.imwrite("logs/pick_conf{}_tool{}.png".format(cur, tid),pick_confs[tid])
			cv2.imwrite("logs/place_conf{}_{}.png".format(cur,angle),place_conf)
			cv2.imwrite("logs/rgb{}.png".format(cur),rgb)
			cv2.imwrite("logs/depth{}.png".format(cur),depth)
		# Pixels to end effector poses.
		hmap = img[:, :, 3]
		p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
		p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
		p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, p0_theta))
		p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, p1_theta))
		# import ipdb; ipdb.set_trace()
		return {
				'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
				'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
				"tool_id": tool_id
		}


class BCTransporterAgent(TransporterAgent):

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)
		self.tool_num = 3
		self.attention = BCnet(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess,
				tool_num=self.tool_num)

	def get_image(self, obs):
		"""Stack color and height images image."""

		# if self.use_goal_image:
		#   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
		#   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
		#   input_image = np.concatenate((input_image, goal_image), axis=2)
		#   assert input_image.shape[2] == 12, input_image.shape

		# Get color and height maps from RGB-D images.
		cmap, hmap = utils.get_fused_heightmap(
				obs, self.cam_config, self.bounds, self.pix_size)
		img = np.concatenate((cmap,
													hmap[Ellipsis, None],
													hmap[Ellipsis, None],
													hmap[Ellipsis, None]), axis=2)
		assert img.shape == self.in_shape, img.shape
		return img

	def get_sample(self, dataset, augment=True):
		"""Get a dataset sample.

		Args:
			dataset: a ravens.Dataset (train or validation)
			augment: if True, perform data augmentation.

		Returns:
			tuple of data for training:
				(input_image, p0, p0_theta, p1, p1_theta)
			tuple additionally includes (z, roll, pitch) if self.six_dof
			if self.use_goal_image, then the goal image is stacked with the
			current image in `input_image`. If splitting up current and goal
			images is desired, it should be done outside this method.
		"""

		(obs, act, _, _), _ = dataset.sample()
		img = self.get_image(obs)

		# Get training labels from data sample.
		p0_xyz, p0_xyzw = act['pose0']
		p1_xyz, p1_xyzw = act['pose1']
		tool_id = act['tid']
		is_mix = act['is_mix']
		# import ipdb;ipdb.set_trace()
		p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
		# use positive??
		p0_theta = np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
		p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
		p1_theta = np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
		p1_theta = p1_theta - p0_theta
		p0_theta = 0
		# Data augmentation.
		if augment:
			img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

		p0_xyz = utils.pix_to_xyz(p0, img[:, :, 3], self.bounds, self.pix_size)
		p1_xyz = utils.pix_to_xyz(p1, img[:, :, 3], self.bounds, self.pix_size)
		return img, p0_xyz, p0_theta, p1_xyz, p1_theta, tool_id, is_mix

	def plot_to_tensor(self,feature,figsize=(3.2,1.6)):
		draw_feat = feature.numpy()
		cmap = plt.get_cmap('inferno')
		figure = plt.figure(figsize=figsize)
		plt.imshow(draw_feat[0],cmap = cmap)
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		plt.close(figure)
		buf.seek(0)

		image = tf.image.decode_png(buf.getvalue(), channels=4)
		# Add the batch dimension
		image = tf.expand_dims(image, 0)
		return image
	def train(self, dataset, writer=None):
		"""Train on a dataset sample for 1 iteration.

		Args:
			dataset: a ravens.Dataset.
			writer: a TF summary writer (for tensorboard).
		"""
		h,w,c = self.in_shape
		tf.keras.backend.set_learning_phase(1)
		img, p0, p0_theta, p1, p1_theta, tool_id, is_mix = self.get_sample(dataset)

		# Get training losses.
		step = self.total_steps + 1
		loss0,feature0 = self.attention.train(img, p0, p0_theta, p1, p1_theta, tool_id = tool_id, is_mix = is_mix)

		
		with writer.as_default():
			sc = tf.summary.scalar

			if step %100 == 0:

				sc('train_loss/attention', loss0, step)
		print(f'Train Iter: {step} Loss: {loss0:.4f} ')
		self.total_steps = step



	def test(self, obs, cur=0, info=None, goal=None, vis=False):  # pylint: disable=unused-argument
		"""Run inference and return best action given visual observations."""
		tf.keras.backend.set_learning_phase(0)
		# Get heightmap from RGB-D images.
		img = self.get_image(obs)
		rgb = img[:,:,:3][:,:,::-1]
		rgb = np.array(rgb, dtype=np.uint8)
		depth = img[:,:,3]
		depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
		depth = np.uint8(depth*255)
		
		# Attention model forward pass.
		# outputs = self.attention.forward(img) # original: 360,720,1. ours: 3,360,720
		# outputs = outputs.numpy()[0]
		# # import ipdb;ipdb.set_trace()
		# p0_pix = (outputs[:2]).astype(int)

		# p1_pix = (outputs[2:4]).astype(int)
		# tool_id = np.argmax(outputs[5:7])
		# p1_theta = 0
		# p0_theta = 0



		outputs = self.attention.forward(img) # original: 360,720,1. ours: 3,360,720
		outputs = outputs.numpy()[0]
		# import ipdb;ipdb.set_trace()
		p0_xyz = (outputs[:3]).astype(int)

		p1_xyz = (outputs[3:6]).astype(int)
		tool_id = np.argmax(outputs[6:9])
		p1_theta = 0
		p0_theta = 0

		p0_pix = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
		p1_pix = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)



		print("predict p0_theta:{0}, picked tool:{1}".format(p0_theta, tool_id))

		print("predict p1_theta:{}",p1_theta*180/np.pi)

		if vis:
			

			rgb = cv2.circle(rgb, p0_pix[::-1], 4, (0,0,255), 1)
			rgb = cv2.circle(rgb, p1_pix[::-1], 4, (0,255,0), 1)

			angle = str(int(p1_theta*180/np.pi))
			cv2.imwrite("logs/rgb{}.png".format(cur),rgb)
			cv2.imwrite("logs/depth{}.png".format(cur),depth)
		# Pixels to end effector poses.
		hmap = img[:, :, 3]
		p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
		p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
		p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, p0_theta))
		p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, p1_theta))
		# import ipdb; ipdb.set_trace()
		return {
				'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
				'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
		}
	def load(self, n_iter):
		"""Load pre-trained models."""
		print(f'Loading pre-trained model at {n_iter} iterations.')
		attention_fname = 'attention-ckpt-%d.h5' % n_iter
		# transport_fname = 'transport-ckpt-%d.h5' % n_iter
		attention_fname = os.path.join(self.models_dir, attention_fname)
		# transport_fname = os.path.join(self.models_dir, transport_fname)
		self.attention.load(attention_fname)
		# self.transport.load(transport_fname)
		self.total_steps = n_iter

	def save(self):
		"""Save models."""
		if not tf.io.gfile.exists(self.models_dir):
			tf.io.gfile.makedirs(self.models_dir)
		attention_fname = 'attention-ckpt-%d.h5' % self.total_steps
		# transport_fname = 'transport-ckpt-%d.h5' % self.total_steps
		attention_fname = os.path.join(self.models_dir, attention_fname)
		# transport_fname = os.path.join(self.models_dir, transport_fname)
		self.attention.save(attention_fname)
		# self.transport.save(transport_fname)
class OriginalTransporterAgent(TransporterAgent):

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)

		self.attention = Attention(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess)
		self.transport = Transport(
				in_shape=self.in_shape,
				n_rotations=self.n_rotations,
				crop_size=self.crop_size,
				preprocess=utils.preprocess)


class NoTransportTransporterAgent(TransporterAgent):

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)

		self.attention = Attention(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess)
		self.transport = Attention(
				in_shape=self.in_shape,
				n_rotations=self.n_rotations,
				preprocess=utils.preprocess)


class PerPixelLossTransporterAgent(TransporterAgent):

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)

		self.attention = Attention(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess)
		self.transport = TransportPerPixelLoss(
				in_shape=self.in_shape,
				n_rotations=self.n_rotations,
				crop_size=self.crop_size,
				preprocess=utils.preprocess)


class GoalTransporterAgent(TransporterAgent):
	"""Goal-Conditioned Transporters supporting a separate goal FCN."""

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)

		self.attention = Attention(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess)
		self.transport = TransportGoal(
				in_shape=self.in_shape,
				n_rotations=self.n_rotations,
				crop_size=self.crop_size,
				preprocess=utils.preprocess)


class GoalNaiveTransporterAgent(TransporterAgent):
	"""Naive version which stacks current and goal images through normal Transport."""

	def __init__(self, name, task, n_rotations=36):
		super().__init__(name, task, n_rotations)

		# Stack the goal image for the vanilla Transport module.
		t_shape = (self.in_shape[0], self.in_shape[1],
							 int(self.in_shape[2] * 2))

		self.attention = Attention(
				in_shape=self.in_shape,
				n_rotations=1,
				preprocess=utils.preprocess)
		self.transport = Transport(
				in_shape=t_shape,
				n_rotations=self.n_rotations,
				crop_size=self.crop_size,
				preprocess=utils.preprocess,
				per_pixel_loss=False,
				use_goal_image=True)
