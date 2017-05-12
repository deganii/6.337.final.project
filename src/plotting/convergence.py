import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
import tensorflow as tf


class PerformancePlotter(object):
    @classmethod
    def plot_accuracy(cls, step_dt =[1,2,3],   train_acc_dt = [0.5,0.7,0.8], test_acc = 0.7):
        fig = plt.figure()
        title="Learning Rate"
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

    @classmethod
    def plot_loss(cls, optim, step_dt =[1,2,3],   train_loss_dt = [0.5,0.7,0.8], test_loss = 0.7):
        fig = plt.figure()
        fig.suptitle('Objective function performance: ' + optim, fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Training Loss')
        #test_acc=0.764000
        ax.plot(step_dt, train_loss_dt, step_dt, [test_loss] * len(step_dt), linewidth=3.0)
        #
        # ax.annotate('Test Accuracy = ' + "{:.1%}".format(test_loss), xy=(100, test_loss), xytext=(150, test_loss-.40),
        #             arrowprops=dict(facecolor='black', shrink=0.05))
        #
        # ax.annotate('Final Training Accuracy = ' + "{:.1%}".format(
        #     train_loss_dt[-1]), xy=(step_dt[-1], train_loss_dt[-1]),
        #             xytext=(700, train_loss_dt[-1]-.20),
        #             arrowprops=dict(facecolor='black', shrink=0.05))
        plt.show()


    @classmethod
    def plot_loss_multi(cls, dataset):
        # load from file
        fig = plt.figure()
        fig.suptitle('Objective function performance: ', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Training Loss')
        ax.set_xlim([0,2.5])
        for optim in dataset.keys():
            data = dataset[optim]
            ax.plot(data['time_dt'], data['train_loss_dt'], linewidth=3.0, label=optim)
            tau = 0.001
            converge_idx = next(i for i, v in enumerate( data['train_loss_dt']) if v < tau)
            converge_time = data['time_dt'][converge_idx]
            converge_val = data['train_loss_dt'][converge_idx]
            ax.annotate('Conv:' + "{0:.1f}s".format(converge_time), xy=(converge_time, converge_val),
                        xytext=(converge_time, 15))

        ax.legend()
        #datasets

        #             arrowprops=dict(facecolor='black', shrink=0.05))
        #
        # ax.annotate('Final Training Accuracy = ' + "{:.1%}".format(
        #     train_loss_dt[-1]), xy=(step_dt[-1], train_loss_dt[-1]),
        #             xytext=(700, train_loss_dt[-1]-.20),
        #             arrowprops=dict(facecolor='black', shrink=0.05))
        plt.show()


    @classmethod
    def plot_accuracy_multi(cls, dataset):
        # load from file
        fig = plt.figure()
        fig.suptitle('Objective function performance: ', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Accuracy')
        #ax.set_xlim([0,30])
        for optim in dataset.keys():
            data = dataset[optim]
            ax.plot(data['time_dt'], data['train_loss_dt'], linewidth=3.0, label=optim)
            tau = 0.001
            converge_idx = next(i for i, v in enumerate( data['train_loss_dt']) if v < tau)
            converge_time = data['time_dt'][converge_idx]
            converge_val = data['train_loss_dt'][converge_idx]
            ax.annotate('Conv:' + "{0:.1f}s".format(converge_time), xy=(converge_time, converge_val),
                        xytext=(converge_time, 15))
        ax.legend()


        #datasets

        #             arrowprops=dict(facecolor='black', shrink=0.05))
        #
        # ax.annotate('Final Training Accuracy = ' + "{:.1%}".format(
        #     train_loss_dt[-1]), xy=(step_dt[-1], train_loss_dt[-1]),
        #             xytext=(700, train_loss_dt[-1]-.20),
        #             arrowprops=dict(facecolor='black', shrink=0.05))
        plt.show()

