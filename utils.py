import tensorflow as tf
import numpy as np
from VGG.vgg19 import Vgg19
import pathlib
import os
from glob import glob
import re
_vgg19 = Vgg19()

# ======================================================== #
# Variable Manager
# ======================================================== #
class VarManager():
	def __init__(self):
		self.variables = dict()
		#tf.GraphKeys.GLOBAL_VARIABLES
		#tf.GraphKeys.TRAINABLE_VARIABLES

	def tic(self, cname, tag=""):
		variables = {
			"var_list": tf.get_collection(cname),
			"cname": cname
		}
		if self.variables.get(tag) == None:
			self.variables[tag] = []
		self.variables[tag].append(variables)

	def toc(self, tag=""):
		if len(self.variables[tag]) == 0 or self.variables.get(tag) == None:
			raise ValueError("No variables in VarManager")
		else:
			last_variables = self.variables[tag].pop()
			out = list(set(tf.get_collection(last_variables["cname"])) - set(last_variables["var_list"]))
			return out

	def ticvar(self, tag=""):
		self.tic(cname=tf.GraphKeys.GLOBAL_VARIABLES, tag=tag)

	def tictrain(self, tag=""):
		self.tic(cname=tf.GraphKeys.TRAINABLE_VARIABLES, tag=tag)

class Model():
	def __init__(self, name=None):
		if name == None:
			self.name = type(self).__name__
		else:
			self.name = name
		self.built = False

	def __call__(self, *args, **kwargs):
		if not self.built:
			varmanager = VarManager()
			varmanager.tictrain("train_var")
			varmanager.ticvar("all_var")

		# run
		with tf.variable_scope(self.name, reuse=self.built):
			out = self.forward(*args, **kwargs)

		if not self.built:
			self.variables = varmanager.toc("all_var")
			self.trainables = varmanager.toc("train_var")

		self.built = True

		return out

	def forward(self, *args, **kwargs):
		pass


# ======================================================== #
# Network util
# ======================================================== #
def get_shape(tensor:tf.Tensor):
	dynamic_shape = tf.shape(tensor)
	static_shape = tensor.get_shape().as_list()
	shape = []
	for i in range(len(static_shape)):
		if static_shape[i] == None:
			shape.append(dynamic_shape[i])
		else:
			shape.append(static_shape[i])
	return shape

def is_static_shape(shape):
	"""
	(1) shape가 list인 경우:
		각 원소가 static_shape인지 dynamic_shape인지 bool로 가지는 list리턴
	(2) shape가 원소 (tf.Tensor or int) 인 경우:
		해당 원소가 static_shape인지 dynamic_shape인지 bool로 리턴
	:param shape:	shape 의 각 원소 혹은 전체
	:return: 		bool or list of bool
	"""
	if isinstance(shape, int):
		return True
	elif isinstance(shape, tf.Tensor):
		return False
	elif isinstance(shape, list):
		out = []
		for e in shape:
			out.append(is_static_shape(e))
		return out
	else:
		raise Exception("layer_tf.is_static_shape: type must be in {int, tf.Tesnor}, but ({})".format(type(shape)))

def instance_norm(out, name="ln"):
	shape = get_shape(out)
	moments_axis = list(np.arange(1, len(shape)-1)) # tf.nn.moments에 사용할 axis, byxc 를 기준으로함
	if len(shape) == 2: # fc layer case
		raise ValueError("layer norm cannot be used for fc layer")
	elif len(shape) == 3: # conv1d case
		b,x,c = shape
	elif len(shape) == 4:  # conv2d case
		b, y, x, c = shape
	elif len(shape) == 5:  # conv3d case
		b, z, y, x, c = shape

	assert is_static_shape(c) == True  # compile time에 채널 디멘젼이 알려져야함

	with tf.variable_scope(name):
		beta = tf.get_variable(initializer=tf.constant(0.0, shape=[c]), name="beta")
		gamma = tf.get_variable(initializer=tf.constant(1.0, shape=[c]), name="gamma")
		batch_mean, batch_var = tf.nn.moments(out, moments_axis, keep_dims=True, name="moments")  # moments_axis내에서만 mean, var을 구한다
		out = tf.nn.batch_normalization(out, batch_mean, batch_var, beta, gamma, 1e-3, name="normed")
	return out

# ======================================================== #
#
# ======================================================== #
def get_vgg_loss(image, label, *args, reducer=lambda input: tf.reduce_mean(tf.abs(input))):
	b = tf.shape(image)[0]
	input = tf.concat([image,label], axis=0)
	_vgg19.build(input)

	out=[]
	for arg in args:
		feature = _vgg19.__getattribute__(arg)
		loss = reducer(feature[:b]-feature[b:])
		out.append(loss)
	return out

class DecayManager:
	def __init__(self, boundaries, values):
		assert len(boundaries)+1==len(values)
		self.boundaries = boundaries
		self.values = values
	def get(self, iter):
		if iter >= self.boundaries[-1]:
			return self.values[-1]
		for i in range(len(self.boundaries)):
			if iter < self.boundaries[i]:
				return self.values[i]

	def __call__(self, iter):
		return self.get(iter)

def get_session_and_run_init(memory_save=False):
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = memory_save
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	return sess

class BatchManager:
	"""
	*args 갯수만큼 dataset들을 들고온다.
	이 때 각 set의 형태는 모두 같다.
	shuffle = True 이면 모두 같은 형태로 shuffle(permutation)해준다.
	next_batch를 통해 정해진 숫자만큼 꺼내주고, 마지막 원소에 도착하면 다시 처음으로 돌아간다.
	"""

	def __init__(self, *args, shuffle=True, is_flip=True):
		self.data = list(args)
		self.setnum = len(self.data)
		self.is_flip = is_flip
		self.flip = 0

		if self.setnum == 0:
			return

		self.batchnum = self.data[0].shape[0]

		for i in range(self.setnum):
			assert self.batchnum== self.data[i].shape[0]


		if shuffle == True:
			self.shuffle_index = np.arange(0, self.batchnum)
			np.random.shuffle(self.shuffle_index)
			for j in range(self.setnum):
				self.data[j] = np.take(self.data[j], self.shuffle_index, axis=0)

		self.index = 0


	def next_batch(self, batch_size):
		assert batch_size <= self.batchnum

		return_data=[]
		if self.index+batch_size > self.batchnum:
			self.flip = self.flip+1
			for i in range(self.setnum):
				fetch_index = list(range(self.index,self.batchnum))
				return_data.append(self.data[i][fetch_index])
				if self.is_flip == True:
					if self.flip%2 == 1:		# i가 홀수면 y flip
						self.data[i] = self.data[i][:, ::-1, :, :]
					else:		# i가 짝수면 x flip
						self.data[i] = self.data[i][:, :, ::-1, :]
				fetch_index = list(range(0, self.index + batch_size - self.batchnum))
				return_data[i] = np.concatenate([return_data[i],self.data[i][fetch_index]])
			self.index = self.index+batch_size-self.batchnum
		else:
			for i in range(self.setnum):
				return_data.append(self.data[i][self.index:self.index+batch_size])
			self.index = self.index + batch_size
		return return_data


# ======================================================== #
# file system
# ======================================================== #
def join(*paths):
	paths = [str(path) for path in paths]
	return str(pathlib.Path(*paths))

def parent(path):
	path = pathlib.Path(path)
	return join(*path.parts[:-1])

def mkdir(path):
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def exist(path):
	return os.path.exists(str(path))

def is_file(path):
	assert exist(path)
	path = pathlib.Path(path)
	return path.is_file()

def is_dir(path):
	assert exist(path)
	path = pathlib.Path(path)
	return path.is_dir()

def readraw(path, dtype):
	if not exist(path) or not is_file(path):
		raise ValueError

	with open(path, "rb") as f:
		data = np.fromfile(f, dtype)
	return data

# ======================================================== #
#	Tensorflow Saver
# ======================================================== #

def latest_idx(path, pattern=r".*-(\d+)"):
	if not exist(path):
		return None

	if not is_dir(path):
		raise ValueError("not a dir")

	idx = -1

	file_list = sorted(glob(join(path, "*.meta")))
	for file in file_list:
		m = re.search(pattern, file)
		if idx < int(m.group(1)):
			idx = int(m.group(1))

	if idx == -1:
		return None
	else:
		return idx

class SaverTF:
	# var_list, file, max_to_keep
	def __init__(self, var_list, path, filename, max_to_keep=5):
		if is_dir(path)==False:
			raise ValueError
		elif exist(path)==False:
			mkdir(path)
		self.path = path
		self.filename = filename
		self.saver = tf.train.Saver(var_list, max_to_keep=max_to_keep)
		pass
	def save(self, sess, global_step):
		while(True):
			try:
				self.saver.save(sess=sess, save_path=join(self.path, self.filename), global_step=global_step)
				break
			except:
				pass
	# i 안넣으면 자동으로 찾아서  로드하거나 그냥 로드 안함

	def load(self, sess, i=None):
		if latest_idx(self.path)==None:
			print("no save file for {}".format(type(self).__name__))
			return 0
		elif i == 0 :
			print("continue without load")
			return 0
		elif i == None:
			i = latest_idx(self.path)
			self.saver.restore(sess, join(self.path, "{}-{}".format(self.filename, i)))
		else:
			self.saver.restore(sess, join(self.path, "{}-{}".format(self.filename, i)))
		return i

	def latest_idx(self):
		return latest_idx(self.path)

# ======================================================== #
# GAN G/D Contorller
# ======================================================== #
class GAN_Ctrl():
	def __init__(self, target_ratio = 0.7):
		self.ng = 1
		self.nd = 1
		self.end_ratio = 20
		self.target_ratio = target_ratio

	def get_nd(self):
		if self.ng <= self.nd:
			nd = self.nd//self.ng
		else:
			nd = 1
		return int(nd)

	def get_ng(self):
		if self.ng <= self.nd:
			ng = 1
		else:
			ng = self.ng//self.nd
		return int(ng)

	def __call__(self, ratio):
		if ratio > self.target_ratio:
			self.nd = self.nd * 1.03
		elif ratio < self.target_ratio:
			self.nd = self.nd * 0.97
		else:
			pass
		# return end flag
		if self.ng/self.nd < 1/self.end_ratio or self.ng/self.nd > self.end_ratio:
			return 1
		else:
			return 0

# ======================================================== #
# image upscaling
# ======================================================== #
def pixel_shuffler(image, n=1):
	if image.shape[0]%n%n !=0:
		raise ValueError
	shape = image.shape
	image = np.reshape(image, [n, n, shape[0]//n//n, shape[1], shape[2], shape[3]])
	image = np.transpose(image, [2, 3, 0, 4, 1, 5])
	image = np.reshape(image, [shape[0]//n//n, shape[1]*n, shape[2]*n, shape[3]])
	return image

def image_upscale(image, n):
	"""
	image : (b,y,x,c) image
	:param image:
	:param n:
	:return:
	"""
	image = np.tile(image, [n**2, 1, 1, 1])
	image = pixel_shuffler(image, n)
	return image