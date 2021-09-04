import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import *


class LiveLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	"""
	Updates learning rate schedule based on config file during the training process.
	"""
	def __init__(self, check_for_update_interval, lr_filename='current_lr.conf', custom_objects={}):
		"""
		:param check_for_update_interval: The interval the lr will be updated. Measured in **steps**.
		:param lr_filename: The file to read config from.
		:param custom_objects:
			The dict passed to `deserialize` function while loading lr schedule.
			This is typically for custom children of tf.keras.optimizers.schedules.LearningRateSchedule.
		"""
		self.lr_filename = lr_filename
		self.check_for_update_interval = check_for_update_interval
		self.custom_objects = custom_objects

		self.lr_file = open(lr_filename, 'r')
		self.base_schedule = None
		self.logger = tf.get_logger()

	def __call__(self, step):
		return tf.numpy_function(self.call, [step], tf.float32)

	def load_config(self):
		try:
			self.lr_file.seek(0)
			config = json.loads(self.lr_file.read())
			self.base_schedule = deserialize(config, self.custom_objects)
		except Exception as err:
			self.logger.error('LiveLrSchedule error: %s', str(err), exc_info=True)
			self.logger.error('LiveLrSchedule keeps schedule unchanged')

	def call(self, step):
		if step % self.check_for_update_interval == 0:
			self.load_config()

		return np.float32(self.base_schedule(step))
