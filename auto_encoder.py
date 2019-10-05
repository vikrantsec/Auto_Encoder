import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from functools import partial

def unpickle(file):

	with open(file, 'rb') as n_file:
		dicti = pickle.load(n_file, encoding='latin1')
	return dicti
	
def grayscale(im):
	return im.reshape(im.shape[0], 3, 32, 32).mean(1).reshape(im.shape[0], -1)
	
data, labels = [], []

for i in range(1,6):
	filename = './cifar-10-batches-py/data_batch_' + str(i)
	open_data = unpickle(filename)
	if len(data) > 0:
		data = np.vstack((data, open_data['data']))
		labels = np.hstack((labels, open_data['labels']))
	else:
		data = open_data['data']
		labels = open_data['labels']
		
data = grayscale(data)
x = np.matrix(data)
y = np.array(labels)

horse_i = np.where(y == 7)[0]

horse_x = x[horse_i]

def plot_image(image, shape=[32,32],cmap='Greys_r'):
	plt.imshow(image.reshape(shape), cmap=cmap, interpolation='nearest')
	plt.show()
	plt.axis('off')
	
plot_image(horse_x[1], shape=[32,32], cmap='Greys_r')

n_inputs = 32*32
BATCH_SIZE = 150
with tf.name_scope('preparation'):
	batch_size = tf.placeholder(tf.int64, name='batch_size')

	x = tf.placeholder(tf.float32, shape=[None, n_inputs], name='x')
	dataset = tf.data.Dataset.from_tensor_slices(x).repeat().batch(batch_size)
	iter = dataset.make_initializable_iterator()
	features = iter.get_next()

	
n_hidden_1 = 300
n_hidden_2 = 150
n_hidden_3 = n_hidden_1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

with tf.name_scope('init_reg'):
	xav_init = tf.contrib.layers.xavier_initializer()
	l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

with tf.name_scope('dense'):

	dense_layer = partial(tf.layers.dense, activation=tf.nn.relu, kernel_initializer=xav_init, kernel_regularizer=l2_regularizer)

	hidden_1 = dense_layer(features, n_hidden_1)
	hidden_2 = dense_layer(hidden_1, n_hidden_2)
	hidden_3 = dense_layer(hidden_2, n_hidden_3)
	outputs = dense_layer(hidden_3, n_outputs, activation=None)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(outputs-features))

with tf.name_scope('optimize'):
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
with tf.name_scope('train'):
	train = optimizer.minimize(loss)

n_batches = horse_x.shape[0] // BATCH_SIZE

n_epochs = 100

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(iter.initializer, feed_dict={x: horse_x, batch_size:BATCH_SIZE})
	writer = tf.summary.FileWriter('./train')
	writer.add_graph(sess.graph)
	
	print('Training...')
	print(sess.run(features).shape)
	for epoch in range(n_epochs):
		for iteration in range(n_batches):
			sess.run(train)
		if epoch%10 == 0:
			loss_train = loss.eval()
			print("\r{}".format(epoch), "Train MSE:", loss_train)
	save_path = saver.save(sess, './model.cpkt')
	
	print('Model saved in path: %s' % save_path)
	
test_data = unpickle('./cifar-10-batches-py/test_batch')
test_x = grayscale(test_data['data'])

plot_image(test_x[10], shape=[32,32], cmap='Greys_r')