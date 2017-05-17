from __future__ import print_function

import numpy as np
import time

import tensorflow as tf
from optimizers import *

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# old values
#ext_grad_learning_rate = 0.01
#adam_learning_rate = 0.001

learning_rate_all = 0.001

ext_grad_learning_rate = learning_rate_all
adam_learning_rate = learning_rate_all

# Parameters
#learning_rate = 0.01
learning_rate = learning_rate_all
training_epochs = 25
batch_size = 100
display_step = 1


#optim = 'ext_grad'
#optim = 'scipy'
optim = 'ext_bfgs'
#optim = 'ext_newton'
#optim = 'adam'

# tf Graph Input
x = tf.placeholder(tf.float32, [batch_size, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [batch_size, 10]) # 0-9 digits recognition => 10 classes

x_var = tf.Variable(x, trainable=False, collections=[],name='x_var',dtype=tf.float32)
y_var = tf.Variable(y, trainable=False, collections=[],name='y_var',dtype=tf.float32)


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
error = tf.matmul(x_var, W) + b
pred = tf.nn.softmax(error) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y_var*tf.log(pred), reduction_indices=1))
# Gradient Descent
if(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(cost)
    optimizer.learning_rate = ext_grad_learning_rate
elif(optim == 'ext_bfgs'):
    optimizer = ExternalBFGSOptimizer(cost)
    optimizer.initialized = False
elif(optim == 'adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate= adam_learning_rate).minimize(cost)
#elif (optim == 'scipy'):
#    optimizer = ScipyOptimizerInterface(cost)
#elif(optim == 'ext_newton'):
#    optimizer = ExternalNewtonOptimizer(error, W)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

train_loss_dt = []
step_dt =[0]
train_accuracy_dt = []
test_accuracy_dt = []

test_loss = 0.0
start_time = time.time()
time_dt = [0]

def saveOutputData(sess):
    sess.run([x_var.initializer, y_var.initializer], feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})

    print("Test Accuracy:", accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]}))
    np.savez('performance_data/mnist/' + optim, step_dt=step_dt, train_loss_dt=train_loss_dt, time_dt=time_dt, test_accuracy_dt=test_accuracy_dt, train_accuracy_dt=train_accuracy_dt)



    # make everything the same size
    for series in [train_loss_dt, time_dt, test_accuracy_dt, train_accuracy_dt]:
        pad = len(step_dt) - len(series)
        if pad > 0:
            np.pad(series, pad, 'constant')

    header = "step_dt, train_loss_dt, time_dt, test_accuracy_dt, train_accuracy_dt"
    stacked = np.stack((step_dt, train_loss_dt, time_dt, test_accuracy_dt, train_accuracy_dt))
    np.savetxt('performance_data/mnist/{0}.txt'.format(optim),np.transpose(stacked), header=header, fmt='%10.15f')


def saveMNISTImagesAndPredictions(iteration, test_data, test_labels, predictions):
    # save a histogram along with image data for select images
    import scipy.misc
    import os
    for i in range(5):
        if not os.path.exists('visual/{0}/'.format(iteration)):
            os.makedirs('visual/{0}/'.format(iteration))

        image = scipy.misc.toimage(test_data[i,:].reshape([28,28]))
        image.save('./visual/{0}/test_{1}.png'.format(iteration, i))

        pred = predictions[i,:]
        gnd_truth = test_labels[i,...]

        import matplotlib.pyplot as plt; plt.rcdefaults()
        import numpy as np
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)

        digits = [i for i in range(0,10)]
        #plt.clf()
        ax.bar(digits, pred, align='center', alpha=0.5)
        ax.set_xticks(digits)
        ax.set_ylabel('Probability')
        ax.set_title('Digit Prediction')
        fig.savefig('visual/{0}/test_hist_{1}.png'.format(iteration, i))
        plt.close(fig)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # compute initial loss

    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run([x_var.initializer, y_var.initializer], feed_dict={x: batch_xs, y: batch_ys})

    c = sess.run( cost, feed_dict={x: batch_xs,y: batch_ys})
    train_loss_dt.append(c)

    train_acc = accuracy.eval({x: batch_xs, y: batch_ys})
    train_accuracy_dt.append(train_acc)
    sess.run([x_var.initializer, y_var.initializer],
             feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})
    test_acc = accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})
    test_accuracy_dt.append(test_acc)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        batch_xs, batch_ys = 0, 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run([x_var.initializer, y_var.initializer], feed_dict={x: batch_xs,
                                                                        y: batch_ys})

            # Run optimization op (backprop) and cost op (to get loss value)
            if optim == 'scipy' or optim == 'ext_grad' or optim == 'ext_bfgs' or optim == 'ext_newton':
                optimizer.minimize(sess, feed_dict={x: batch_xs, y: batch_ys})
                c = sess.run( cost, feed_dict={x: batch_xs,y: batch_ys})
                train_loss_dt.append(c)
                #print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                #print("Test Accuracy:", test_acc)
            elif (optim == 'adam'):
                sess.run(optimizer)
                c = sess.run(cost, feed_dict={x: batch_xs,y: batch_ys})
                train_loss_dt.append(c)


            step_dt.append(i)
            elapsed_time = time.time() - start_time
            time_dt.append(elapsed_time)

            if(optim != 'ext_bfgs'):
                train_acc = accuracy.eval({x: batch_xs, y: batch_ys})
                #print("Train Accuracy:",train_acc)
                train_accuracy_dt.append(train_acc)
                sess.run([x_var.initializer, y_var.initializer],
                         feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})

                test_acc = accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})
                test_accuracy_dt.append(test_acc)

            # Compute average loss
            avg_cost += c / total_batch
            if(optim == 'ext_bfgs'):
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

                #train_loss_dt.append(c)
                sess.run([x_var.initializer, y_var.initializer], feed_dict={x: batch_xs,
                                                                        y: batch_ys})


                train_acc = accuracy.eval({x: batch_xs, y: batch_ys})
                print("Train Accuracy:", train_acc)
                train_accuracy_dt.append(train_acc)

                sess.run([x_var.initializer, y_var.initializer],
                         feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})

                #test_acc =accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})
                test_acc =accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})
                test_accuracy_dt.append(test_acc)
                print("Test Accuracy:", test_acc)
                #saveOutputData(sess)

                sess.run([x_var.initializer, y_var.initializer],
                         feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})
                test_preds = pred.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})


                #saveMNISTImagesAndPredictions(i, mnist.test.images, mnist.test.labels, test_preds)

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            sess.run([x_var.initializer, y_var.initializer], feed_dict={x: batch_xs,
                                                                        y: batch_ys})

            print("Train Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))

            sess.run([x_var.initializer, y_var.initializer],
                     feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]})

            print("Test Accuracy:",
                  accuracy.eval({x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size]}))



    print("Optimization Finished!")
    #saveOutputData(sess)

