import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
import tensorflow as tf



fig = plt.figure()
title="Double Learning Rate"
fig.suptitle('Training Accuracy: ' + title, fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_xlabel('Steps')
ax.set_ylabel('Training Accuracy')
#test_acc=0.764000
ax.plot(step_dt, train_acc_dt, step_dt, [test_acc] * len(step_dt), linewidth=3.0)

ax.annotate('Test Accuracy = ' + "{:.1%}".format(test_acc), xy=(100, test_acc), xytext=(150, test_acc-.40),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Final Training Accuracy = ' + "{:.1%}".format(
            train_acc_dt[-1]), xy=(step_dt[-1], train_acc_dt[-1]),
            xytext=(700, train_acc_dt[-1]-.20),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()