
import tensorflow as tf
from optimizers import *
import time
import numpy as np

# Set model weights
W = tf.Variable([[1.,2.,3.],[2.,-1.,0.],[3.,1.,-1.]],trainable=False,dtype=tf.float32, name='jacobian')
#W = tf.Variable([[1.,2.,3.],[2.,-1.,1.],[3.,0.,-1.]],trainable=False,dtype=tf.float32)
b = tf.Variable([9.,8.,3.],trainable=False,dtype=tf.float32)
x = tf.Variable([[1.,1.,1.]],dtype=tf.float32 )
# Construct model
error = (tf.matmul(x, W) - b)

# Minimize error using cross entropy
loss = tf.reduce_mean(tf.reduce_sum(tf.square(error)))

# Gradient Descent
#optim = 'ext_grad'
# BFGS
optim = 'ext_bfgs'
#optim = 'ext_newton'

#optim = 'ext_d_simplex'
# optim = 'adam'
learning_rate = 0.001
ext_grad_learning_rate = 0.001
adam_learning_rate = 0.001

if(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(loss)
    optimizer.learning_rate = ext_grad_learning_rate
elif(optim == 'ext_bfgs'):
    optimizer = ExternalBFGSOptimizer(loss)
    optimizer.initialized = False
# #elif(optim == 'ext_newton'):
#     # for this problem, the weight matrix W *is* the jacobian at all values
#     optimizer = ExternalNewtonOptimizer(error, W)
# elif(optim == 'ext_d_simplex'):
#     # for this problem, the weight matrix W *is* the jacobian at all values
#     optimizer = ExternalDownhillSimplexOptimizer(loss)
elif(optim == 'adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate= adam_learning_rate).minimize(loss)

init = tf.global_variables_initializer()
trainingIter = 20000
i = 0
idx = 0
step_dt =[0]
train_loss_dt = []
acc_dt = []

test_loss = 0.0
start_time = time.time()
time_dt = [0]


with tf.Session() as session:
    session.run(init)

    train_loss_dt.append(session.run(loss))

    if (optim == 'adam'):
        while(i < trainingIter):
            session.run(optimizer)
            i = i+1
            if (i % 1 == 0):
                loss_res = session.run(loss)
                train_loss_dt.append(loss_res)
                step_dt.append(i)
                elapsed_time = time.time() - start_time
                time_dt.append(elapsed_time)
                finalX = session.run(x)
                print(finalX)
                if loss_res < 1e-12:
                    break

    else:
        #for  i in range(10000):
        #    session.run(optimizer)
        #feed_dict = {input: np.}

        for i in range(trainingIter):
            optimizer.minimize(session)
            loss_res = session.run(loss)
            train_loss_dt.append(loss_res)
            step_dt.append(i)
            elapsed_time = time.time() - start_time
            time_dt.append(elapsed_time)

            #errorVal = session.run(error)
            finalX = session.run(x)
            print(finalX)
            #print(errorVal)
            if loss_res < 1e-12:
                break

#np.savez('performance_data/toy/' + optim, step_dt=step_dt, train_loss_dt=train_loss_dt, time_dt=time_dt)
#header = 'step_dt,time_dt,train_loss_dt'

#stacked = np.stack((step_dt,time_dt,train_loss_dt))
#np.savetxt('performance_data/toy/{0}.txt'.format(optim),np.transpose(stacked), header=header, fmt='%10.15f')

# plot and show
#from plotting import PerformancePlotter
#PerformancePlotter.plot_loss(optim, step_dt, train_loss_dt, test_loss )


# show time per iteration
# time to convergence
# steps to convergence
# training / test accuracy for each method

