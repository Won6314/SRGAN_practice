import matplotlib.pyplot as plt
import numpy as np
import imageio
from utils import mkdir, exist, parent

def imwrite(path, data, data_format="byxc"):
	if not exist(parent(path)):
		mkdir(parent(path))
	if data.ndim != 4:
		raise ValueError("iomanager.imwrite: image's dimension is not 4d")

	if data_format.lower() == "byxc":
		pass
	elif data_format.lower() == "bcyx":
		data = np.transpose(data, (0,3,1,2))
	else:
		raise ValueError('iomanager.imwrite: this data format is not supported. format:({})' .format(data_format))

	b, y, x, c = data.shape

	if b!=1 or (c!=1 and c!=3 and c!=4):
		raise ValueError("iomanager.imwrite: not appropriate b,y,x,c ({},{},{},{})".format(b,y,x,c))

	if c ==3 or c==4:
		img = np.clip((data[0]*255), 0, 255).astype(np.uint8)
	elif c==1:
		img = np.clip((data[0,:,:,0]*255), 0, 255).astype(np.uint8)
	imageio.imsave(path, img)


def show(img, path=None):
	if path==None:
		plt.figure()
		if img.shape[2]==1:
			fig = plt.imshow(img[:,:,0], cmap="gray")
		elif img.shape[2] in [3,4]:
			fig = plt.imshow(img)
		else:
			raise ValueError
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.pause(0.001)
		plt.show(block=False)
		plt.tight_layout()
	else:
		imwrite(path, img[None,:,:,:])

def normalize_image(img: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
	if vmin == None:
		vmin = img.min()
	if vmax == None:
		vmax = img.max()
	img = (img-vmin)/(vmax-vmin)
	img = np.clip(img, 0, 1).astype(np.float32)
	return img



def nshow(img, x_num, data_format="byxc", is_color=True, vmin=None, vmax=None, path=None):
	# ======================================================== #
	# if float32 input & no vmin/vmax -> vmin/vmax = 0/1
	# ======================================================== #
	if img.dtype==np.float32:
		if vmin==None:
			vmin=0
		if vmax==None:
			vmax=1

	img = normalize_image(img, vmin, vmax)
	# is_color = gray or 3~4ch
	if img.ndim != 4:
		raise ValueError("imshow.nshow : image demension is not 4d")

	if data_format == "byxc":
		pass
	elif data_format == "bcyx":
		img = np.transpose(img,(0,3,1,2))
	else:
		raise ValueError("imshow.nshow : data_format is not supported")

	if is_color:
		if not (img.shape[3] in [1,3,4]):
			img = np.transpose(img , (0,3,1,2))
			img = np.reshape(img, [img.shape[0]*img.shape[1],img.shape[2],img.shape[3],1])
			nshow(img, x_num, is_color=is_color, path=path)

		else:
			if img.shape[0] % x_num != 0:
				empty_img = np.zeros([x_num - img.shape[0] % x_num, img.shape[1], img.shape[2], img.shape[3]], dtype=img.dtype)
				img = np.concatenate([img, empty_img], axis=0)

			merged_image = []
			for i in range((img.shape[0] - 1) // x_num + 1):
				x_merged_image = []
				for j in range(x_num):
					x_merged_image.append(img[j + i * x_num])
				x_merged_image = np.concatenate(x_merged_image, axis=1)
				merged_image.append(x_merged_image)
			merged_image = np.concatenate(merged_image, axis=0)
			if img.shape[3] in [1,3,4]:
				show(merged_image, path= path)
	elif is_color == False:
		if img.shape[3]!= 1:
			img = np.transpose(img, (0, 3, 1, 2))
			img = np.reshape(img, [img.shape[0] * img.shape[1], img.shape[2], img.shape[3], 1])
			nshow(img, x_num, is_color=is_color)
		else:
			if img.shape[0] % x_num != 0:
				empty_img = np.zeros([x_num - img.shape[0] % x_num, img.shape[1], img.shape[2], img.shape[3]], dtype=img.dtype)
				img = np.concatenate([img, empty_img], axis=0)

			merged_image = []
			for i in range((img.shape[0] - 1) // x_num + 1):
				x_merged_image = []
				for j in range(x_num):
					x_merged_image.append(img[j + i * x_num])
				x_merged_image = np.concatenate(x_merged_image, axis=1)
				merged_image.append(x_merged_image)
			merged_image = np.concatenate(merged_image, axis=0)
			show(merged_image, path=path)
	else:
		raise ValueError("imshow.nshow: image color is wrong")