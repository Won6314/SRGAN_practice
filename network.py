from set1.SRGAN.utils import Model, get_shape, instance_norm
import tensorflow.contrib.slim as slim
import tensorflow as tf


def tf_pixel_shuffler(image, n=1):
	shape = get_shape(image)
	# if shape[0]%(n**2) != 0:
	# 	raise ValueError
	image = tf.reshape(image, [n, n, shape[0]//n//n, shape[1], shape[2], shape[3]])
	image = tf.transpose(image, [2, 3, 0, 4, 1, 5])
	image = tf.reshape(image, [shape[0]//n//n, shape[1]*n, shape[2]*n, shape[3]])
	return image

def tf_image_upscale(image, n):
	image = tf.tile(image, [n**2, 1, 1, 1])
	image = tf_pixel_shuffler(image, n)
	return image

def Prelu(x):
	alphas = tf.get_variable('alpha', x.get_shape()[-1],
							 initializer=tf.constant_initializer(0.25),
							 dtype=tf.float32)
	return tf.maximum(x, x*alphas)

def Gen_block(input, name="gen_block"):
	out = input
	with tf.variable_scope(name):
		out = slim.conv2d(out, 64, 3, 1, scope="conv1", activation_fn=None)
		out = instance_norm(out, "instance_norm1")
		out = Prelu(out)
		out = slim.conv2d(out, 64, 3, 1, scope="conv2", activation_fn=None)
		out = instance_norm(out, "instance_norm2")
		out = out + input
	return out

def Gen_upscale_block(input, name="gen_up_block"):
	out = input
	with tf.variable_scope(name):
		out = slim.conv2d(out, 256, 3, 1, scope="conv1", activation_fn=None)
		out = tf_image_upscale(out, 2)
		out = Prelu(out)
	return out

def Dis_block(input, n, k, s,  name="dis_block"):
	out = input
	with tf.variable_scope(name):
		out = slim.conv2d(out, n, k, s, scope="conv1", activation_fn=None)
		out = instance_norm(out, "instance_norm1")
		out = tf.nn.leaky_relu(out)
	return out


class Gen_srgan(Model):
	def forward(self, input):
		out = input
		out = slim.conv2d(out, 64, 9, 1, scope="conv1", activation_fn=None)
		out = Prelu(out)
		residual = out
		out = Gen_block(out, "block1")
		out = Gen_block(out, "block2")
		out = Gen_block(out, "block3")
		out = Gen_block(out, "block4")
		out = Gen_block(out, "block5")
		out = slim.conv2d(out, 64, 3, 1, scope="conv2", activation_fn=None)
		out = instance_norm(out)
		out = out + residual
		out = Gen_upscale_block(out, "up_block1")
		out = Gen_upscale_block(out, "up_block2")
		out = slim.conv2d(out, 3, 9, 1, scope="conv3", activation_fn=None)
		return out

class Dis_srgan(Model):
	def forward(self, input):
		out = input
		out = slim.conv2d(out, 64, 3, 1, scope="conv1", activation_fn=tf.nn.leaky_relu)
		out = Dis_block(out, 64, 3, 2, name="block1")
		out = Dis_block(out, 128, 3, 1, name="block2")
		out = Dis_block(out, 128, 3, 2, name="block3")
		out = Dis_block(out, 256, 3, 1, name="block4")
		out = Dis_block(out, 256, 3, 2, name="block5")
		out = Dis_block(out, 512, 3, 1, name="block6")
		out = Dis_block(out, 512, 3, 2, name="block7")
		out = slim.flatten(out)
		out = slim.fully_connected(out, 1024, activation_fn=tf.nn.leaky_relu)
		out = slim.fully_connected(out, 1, activation_fn=None)
		return out
