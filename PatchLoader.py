from glob import glob
import numpy as np
from utils import join, readraw
from tqdm import tqdm

class PatchLoader():
	def __init__(self, path, file_num, lr_size=96, upscale=3, dtype=np.uint8, rgb=1):
		hr_size = lr_size * upscale
		self.file_num = file_num
		path = path

		input_list = sorted(glob(join(path, "input_*.raw")))
		label_list = sorted(glob(join(path, "label_*.raw")))
		permuted_list = np.arange(0, len(input_list))
		np.random.shuffle(permuted_list)
		self.input = list()
		self.label = list()
		for i in tqdm(range(file_num)):
			input = readraw(input_list[permuted_list[file_num]], dtype=dtype)
			label = readraw(label_list[permuted_list[file_num]], dtype=dtype)
			input = np.reshape(input, [-1, lr_size, lr_size, 3])
			label = np.reshape(label, [-1, hr_size, hr_size, 3])
			if rgb == 0:
				input = input[:, :, :, ::-1]
				label = label[:, :, :, ::-1]

			self.input.append(input)
			self.label.append(label)
		self.input = np.concatenate(self.input, axis=0)
		self.label = np.concatenate(self.label, axis=0)


def normalize(image):
	image = (image / 255).astype(np.float32)
	return image


def denormalize(image):
	image = (image * 255).astype(np.uint8)
	return image
