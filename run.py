import argparse
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf

from network import *
from PatchLoader import PatchLoader

from utils import *
from imshow import nshow
# ======================================================== #
#
#
#
#
# ======================================================== #


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--eval_period', '-ep', type=int, default=10)
parser.add_argument('--save_period', '-sp', type=int, default=100)
parser.add_argument('--num_iter', '-ne', type=int, default=100000)
parser.add_argument('--network_name', '-nn', type=str, default='SR_GAN_24')
parser.add_argument('--decay_boundaries', '-db', type=int, nargs='+', default=[10000, 20000, 35000, 100000, 150000])
parser.add_argument('--decay_value', '-dv', type=int, nargs='+', default=[3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
parser.add_argument('--path_root', '-pr', type=str, default=r'C:\LocalData')

args = parser.parse_args([])

batch_size = args.batch_size
eval_period = args.eval_period
save_period = args.save_period
num_iter = args.num_iter
network_name = args.network_name

path_root = join(args.path_root, "set1/SRGAN/{}".format(network_name))
path_log = join(path_root, "log")
path_model = join(path_root, "model")
path_tensorboard = join(path_root, "tensorboard")
path_data = join(args.path_root, r'DIV2K_train_patch_24')
G_path = join(path_log, "G_model")
D_path = join(path_log, "D_model")
mkdir(path_log)
mkdir(path_model)
mkdir(path_tensorboard)
mkdir(G_path)
mkdir(D_path)
decaymanager = DecayManager(args.decay_boundaries, args.decay_value)
# ======================================================== #
#
# ======================================================== #
tf.reset_default_graph()
input = tf.placeholder(tf.float32, [batch_size, 24, 24, 3], "input")
label = tf.placeholder(tf.float32, [batch_size, 96, 96, 3], "label")
learning_rate = tf.placeholder(tf.float32)
G_net = Gen_srgan()
D_net = Dis_srgan()

out = G_net(input)
D_fake = D_net(out)
D_fake_sig = tf.sigmoid(D_fake)
D_real = D_net(label)
D_real_sig = tf.sigmoid(D_real)
ratio = tf.reduce_mean(D_fake_sig)/tf.reduce_mean(D_real_sig)
l1_loss = tf.reduce_mean(tf.abs(out - label))*500
vgg_losses = get_vgg_loss(out, label, 'conv4_4')
vgg_loss = vgg_losses[0]*1800
G_sigmoid_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones([batch_size, 1])))/6
G_loss = l1_loss + vgg_loss + G_sigmoid_loss


D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones([batch_size, 1]))\
						+ tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros([batch_size, 1])))

G_apply_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=G_net.trainables)
D_apply_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=D_net.trainables)

sess = get_session_and_run_init(True)
patchloader = PatchLoader(path_data, 50, 24, 4, dtype=np.float32)
batchmanager = BatchManager(patchloader.input, patchloader.label)

G_saver = SaverTF(G_net.variables, G_path, 'G_model', 1000)
D_saver = SaverTF(D_net.variables, D_path, 'D_model', 1000)

loaded_index = G_saver.load(sess)
D_saver.load(sess, loaded_index)


gan_ctrl = GAN_Ctrl(0.5)
train = edict()
for i in range(loaded_index, loaded_index+num_iter):
	if i % eval_period == 0:
		train.input, train.label = batchmanager.next_batch(16)
		train.D_fake, train.D_real, train.ratio, train.out = sess.run([D_fake_sig, D_real_sig, ratio, out], {input: train.input, label: train.label})
		train.l1, train.vgg_loss, train.G_sigmoid_loss = sess.run([l1_loss, vgg_loss, G_sigmoid_loss], {input: train.input, label: train.label})
		print("[train] [{}] D_fake:[{}] D_real:[{}] ratio:[{}]".format(i, np.mean(train.D_fake), np.mean(train.D_real), train.ratio))
		# print("[train] G_iter:[{}] D_iter:[{}]".format(gan_ctrl.get_ng()*30, gan_ctrl.get_nd()))
		print("[train] L1_loss [{}] G_sigmoid_loss [{}] vgg_loss [{}]".format(train.l1, train.G_sigmoid_loss, train.vgg_loss))
		nshow(np.concatenate([image_upscale(train.input, 4), train.out, train.label]), x_num=16, path=join(path_log, "{}/{:06d}.jpg".format(network_name, i)), vmin=0, vmax=1)
		end_train = gan_ctrl(train.ratio)
		if end_train == 1:
			print("D_iter/G_iter is over 7 or 1/7. end train")
			break

	if i % save_period == save_period-1:
		G_saver.save(sess, i+1)
		D_saver.save(sess, i+1)

	if batchmanager.flip == 4:
		batchmanager.flip = 0
		patchloader = PatchLoader(path_data, 50, 24, 4, dtype=np.float32)
		batchmanager = BatchManager(patchloader.input, patchloader.label)

	for g in range(gan_ctrl.get_ng()):
		train.input, train.label = batchmanager.next_batch(batch_size)
		sess.run([G_apply_op], {input: train.input, label: train.label, learning_rate: decaymanager(i)})
	for d in range(gan_ctrl.get_nd()):
		train.input, train.label = batchmanager.next_batch(batch_size)
		sess.run([D_apply_op], {input: train.input, label: train.label, learning_rate: decaymanager(i)})