'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from optimizers import *

# Uncomment whichever optimizer you want to use
#optim = 'grad'
#optim = 'quasi'
# optim = 'ext_grad'
# optim = 'native_grad'
# optim = 'Adam_Opt'
optim = 'scipy'
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
grad_learning_rate = 0.0001
adam_learning_rate = 0.001
ext_grad_learning_rate = 0.0001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 1.0 #0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, n_input],name='x_placeholder')
y = tf.placeholder(tf.float32, [batch_size, n_classes],name='y_placeholder')
x_var = tf.Variable(x, trainable=False, collections=[],name='x_var',dtype=tf.float32)
y_var = tf.Variable(y, trainable=False, collections=[],name='y_var',dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32,name='dropout_prob') #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x_var, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_var))

#  modification
if(optim == 'grad'):
    optimizer = PurePythonGradientDescentOptimizer(learning_rate=grad_learning_rate).minimize(cost)
elif(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(cost)
    optimizer.learning_rate = ext_grad_learning_rate
elif(optim == 'native_grad'):
    optimizer = tf.train.GradientDescentOptimizer(grad_learning_rate).minimize(cost)
elif(optim == 'Adam_Opt'):
    optimizer = tf.train.AdamOptimizer(learning_rate= adam_learning_rate).minimize(cost)
elif(optim == 'scipy'):
    optimizer = ScipyOptimizerInterface(cost)
else:
    optimizer = QuasiNewton().minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_var, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run([x_var.initializer, y_var.initializer], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        # Run optimization op (backprop)
        if optim == 'scipy' or optim == 'ext_grad':
            #for i in range(10):

                #sess.run(y_var.initializer)
                #load the data
                #sess.run([x_var.initializer,y_var.initializer], feed_dict={x: batch_x, y:batch_y, keep_prob: 1.})
            optimizer.minimize(sess,  feed_dict={x: batch_x, y:batch_y, keep_prob: 1.})
        else:
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy

            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))