'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from optimizers import *

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

ext_grad_learning_rate = 0.01

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1


#optim = 'ext_grad'
#optim = 'scipy'
optim = 'ext_bfgs'
# tf Graph Input
x = tf.placeholder(tf.float32, [batch_size, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [batch_size, 10]) # 0-9 digits recognition => 10 classes

x_var = tf.Variable(x, trainable=False, collections=[],name='x_var',dtype=tf.float32)
y_var = tf.Variable(y, trainable=False, collections=[],name='y_var',dtype=tf.float32)


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x_var, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y_var*tf.log(pred), reduction_indices=1))
# Gradient Descent
if(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(cost)
    optimizer.learning_rate = ext_grad_learning_rate
elif (optim == 'scipy'):
    optimizer = ScipyOptimizerInterface(cost)
elif(optim == 'ext_bfgs'):
    optimizer = ExternalBFGSOptimizer(cost)
    optimizer.initialized = False
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)


    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run([x_var.initializer, y_var.initializer], feed_dict={x: batch_xs,
                                                                        y: batch_ys})


            # Run optimization op (backprop) and cost op (to get loss value)
            if optim == 'scipy' or optim == 'ext_grad' or optim == 'ext_bfgs':
                optimizer.minimize(sess, feed_dict={x: batch_xs, y: batch_ys})
                c = sess.run( cost, feed_dict={x: batch_xs,y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print("Train Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print("Train Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))

    print("Optimization Finished!")

    sess.run([x_var.initializer, y_var.initializer], feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})

    print("Test Accuracy:", accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]}))