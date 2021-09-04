import json

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import *


def import_lr_schedule(config: dict) -> tf.keras.optimizers.schedules.LearningRateSchedule:
	cls = globals()[config['class_name']]
	return cls(**config['params'])


class LiveLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	"""
	Updates learning rate schedule based on config file during the training process.
	"""
	def __init__(self, check_for_update_interval, lr_filename='current_lr.conf'):
		"""
		:param check_for_update_interval: The interval the lr will be updated. Measured in **steps**.
		:param lr_filename: The file to read config from.
		"""
		self.lr_filename = lr_filename
		self.check_for_update_interval = check_for_update_interval

		self.lr_file = open(lr_filename, 'r')
		self.base_schedule = import_lr_schedule(json.loads(self.lr_file.read()))
		self.logger = tf.get_logger()

	def __call__(self, step):
		if step % self.check_for_update_interval == 0:
			try:
				self.lr_file.seek(0)
				config = json.loads(self.lr_file.read())
				self.base_schedule = import_lr_schedule(config)
			except Exception as err:
				self.logger.error('LiveLrSchedule error: {}'.format(str(err)))
				self.logger.error('LiveLrSchedule keeps schedule unchanged')

		return self.base_schedule(step)
