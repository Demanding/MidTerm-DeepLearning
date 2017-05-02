import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
NUM_CLASSES = 102
TEMP_SOFTMAX = 5.0
VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

                self.data_dict = np.load("bvlc_alexnet.npy").item()


        def conv(self,input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group = 1):
                c_i = input.get_shape()[-1]
                convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
                if group==1:
                    conv = convolve(input, kernel)
                else:
                    input_groups =  tf.split(input, group, 3)   
                    kernel_groups = tf.split(kernel, group, 3)
                    output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                    conv = tf.concat(output_groups, 3)
                    
                return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


	def build(self, rgb, train_mode=None):
                rgb = tf.image.resize_images(rgb, [227,227])
		# conv1_1
		with tf.name_scope('conv1_1') as scope:
			kernel = tf.Variable(self.data_dict["conv1"][0], name='weights')
			 
			biases = tf.Variable(self.data_dict["conv1"][1], name='biases')
                        out = self.conv(rgb, kernel, biases, 11, 11, 96, 4, 4, padding="SAME", group=1)
			#out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv1_1 = tf.nn.relu(out, name=scope)
                        #pdb.set_trace()

			self.parameters += [kernel, biases]
			
		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 3, 3, 1],
									strides=[1, 2, 2, 1],
									padding='VALID',
									name='pool1')
                #pdb.set_trace()
                #conv2_1
		with tf.name_scope('conv2_1') as scope:
			kernel = tf.Variable(self.data_dict["conv2"][0],name='weights')
			biases = tf.Variable(self.data_dict["conv2"][1], name='biases')
                 #       pdb.set_trace()
			#out = tf.nn.bias_add(conv, biases)
                        out = self.conv(self.pool1, kernel, biases, 5,5, 256, 1, 1, padding="SAME", group=2)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv2_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 3, 3, 1],
									strides=[1, 2, 2, 1],
									padding='VALID',
									name='pool2')
                
                #conv3_1
		with tf.name_scope('conv3_1') as scope:
			kernel = tf.Variable(self.data_dict["conv3"][0] ,name='weights')
			biases = tf.Variable(self.data_dict["conv3"][1], name='biases')
                        out = self.conv(self.pool2, kernel, biases, 3,3, 384, 1, 1, padding="SAME", group=1)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv3_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv4_1') as scope:
			kernel = tf.Variable(self.data_dict["conv4"][0] ,name='weights')
			biases = tf.Variable(self.data_dict["conv4"][1], name='biases')
                        out = self.conv(self.conv3_1, kernel, biases, 3,3, 384, 1,1, padding="SAME", group=2)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv4_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]
                                                                 
		with tf.name_scope('conv5_1') as scope:
			kernel = tf.Variable(self.data_dict["conv5"][0] ,name='weights')
			biases = tf.Variable(self.data_dict["conv5"][1], name='biases')
                        out = self.conv(self.conv4_1, kernel, biases, 3, 3, 256, 1, 1, padding="SAME", group=2)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv5_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]
		
                self.pool3 = tf.nn.max_pool(self.conv5_1,
									ksize=[1, 3, 3, 1],
									strides=[1, 2, 2, 1],
									padding='VALID',
									name='pool3')
                # fc1
		with tf.name_scope('fc1') as scope:
                        shape = int(np.prod(self.pool3.get_shape()[1:]))
                        pool3_flat = tf.reshape(self.pool3, [-1, shape])
			fc1w = tf.Variable(self.data_dict["fc6"][0], name='weights')
			fc1b = tf.Variable(self.data_dict["fc6"][1], trainable=True, name='biases')
			fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
                        mean, var = tf.nn.moments(fc1l, axes=[0])
                        batch_norm = (fc1l - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.fc1 = tf.nn.relu(batch_norm)
                        #self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc1w, fc1b]
            
		with tf.name_scope('fc2') as scope:
			fc2w = tf.Variable(self.data_dict["fc7"][0],
					 trainable = True,name='weights')
			fc2b = tf.Variable(self.data_dict["fc7"][1],trainable=True, name='biases')
			fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                        mean, var = tf.nn.moments(fc2l, axes=[0])
                        batch_norm = (fc2l - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.fc2 = tf.nn.relu(batch_norm)
                        #self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc2w, fc2b]
            
            
		with tf.name_scope('fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, NUM_CLASSES],
														 dtype=tf.float32, stddev=1e-2), trainable = True,name='weights')
			fc3b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								 trainable=True, name='biases')
			
			fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                        mean, var = tf.nn.moments(fc3l, axes=[0])
                        batch_norm = (fc3l - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.fc3 = tf.nn.relu(batch_norm)
                        #self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc3w, fc3b]
            

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

        def train_last_layer_variables(self, train_last_layer_variables):
                train_last_layer_variables.append([v for v in tf.global_variables() if v.name == "fc3/weights:0"][0])
                train_last_layer_variables.append([v for v in tf.global_variables() if v.name == "fc3/biases:0"][0])
                return train_last_layer_variables


	def training(self, loss, learning_rate, learning_rate_pretrained, train_last_layer_variables, variables_to_restore):
		tf.summary.scalar('loss', loss)
                optimizer1 = tf.train.AdamOptimizer(learning_rate_pretrained)
		optimizer2 = tf.train.AdamOptimizer(learning_rate)
		
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
                train_op1 = optimizer1.minimize(loss, global_step= self.global_step, var_list = variables_to_restore)
		train_op2 = optimizer2.minimize(loss, global_step=self.global_step, var_list = train_last_layer_variables)
		

		return tf.group(train_op1, train_op2)
