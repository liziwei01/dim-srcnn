"""
Author: liziwei01
Date: 2021-12-12 01:42:22
LastEditors: liziwei01
LastEditTime: 2021-12-12 02:07:49
Description: file content
"""
import os
import h5py
import numpy as np
import tensorflow as tf
import time
import scipy.misc
import scipy.ndimage
from tensorflow.python.training import optimizer
from prepare import DataOperations

class Trainer:
    ReLU = "ReLU"

    __trainable = True
    __normalStrides = [1,1,1,1]

    def __init__(self, do=DataOperations(), epoch=150000, batchSize=128, cDim=1, padding="VALID"):
        self.DO = do
        self.__epoch = epoch
        self.__batchSize = batchSize
        self.CDim = cDim
        self.__padding = padding

        # self.DO.PrepareTrainingData()

        tf.compat.v1.disable_eager_execution()
        self.Images = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.CDim], name="images")
        self.Labels = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.CDim], name="labels")

    def GetSRCNNWeights(self):
        weights = {
            "w1": tf.Variable(initial_value=tf.random.normal([9, 9, 1, 64], stddev=1e-3), trainable=self.__trainable, name="w1"),
            "w2": tf.Variable(initial_value=tf.random.normal([1, 1, 64, 32], stddev=1e-3), trainable=self.__trainable, name="w2"),
            "w3": tf.Variable(initial_value=tf.random.normal([5, 5, 32, 1], stddev=1e-3), trainable=self.__trainable, name="w3")
        }
        return weights

    def GetSRCNNBiases(self):
        biases = {
            "b1": tf.Variable(initial_value=tf.zeros([64]),trainable=self.__trainable ,name="b1"),
            "b2": tf.Variable(initial_value=tf.zeros([32]),trainable=self.__trainable, name="b2"),
            "b3": tf.Variable(initial_value=tf.zeros([1]),trainable=self.__trainable, name="b3")
        }
        return biases

    def GetSRCNNOptimizer(self):
        optimizer = {
            "o1": tf.compat.v1.train.GradientDescentOptimizer(1e-4),
            "o2": tf.compat.v1.train.GradientDescentOptimizer(1e-4),
            "o3": tf.compat.v1.train.GradientDescentOptimizer(1e-5)
        }
        return optimizer

    def GetDimSRCNNWeights(self):
        weights = {
            "w1": tf.Variable(initial_value=tf.random.normal([5, 5, 1, 32], stddev=1e-3), trainable=self.__trainable, name="w1"),
            "w2": tf.Variable(initial_value=tf.random.normal([1, 1, 32, 16], stddev=1e-3), trainable=self.__trainable, name="w2"),
            "w3": tf.Variable(initial_value=tf.random.normal([3, 3, 16, 1], stddev=1e-3), trainable=self.__trainable, name="w3")
            # "w1": tf.Variable(initial_value=tf.random.normal([13, 13, 1, 128], stddev=1e-3), trainable=self.__trainable, name="w1"),
            # "w2": tf.Variable(initial_value=tf.random.normal([7, 7, 128, 1], stddev=1e-3), trainable=self.__trainable, name="w2")
        }
        return weights

    def GetDimSRCNNBiases(self):
        biases = {
            "b1": tf.Variable(initial_value=tf.zeros([32]), trainable=self.__trainable, name="b1"),
            "b2": tf.Variable(initial_value=tf.zeros([16]), trainable=self.__trainable, name="b2"),
            "b3": tf.Variable(initial_value=tf.zeros([1]), trainable=self.__trainable, name="b3")
            # "b1": tf.Variable(initial_value=tf.zeros([128]), trainable=self.__trainable, name="b1"),
            # "b2": tf.Variable(initial_value=tf.zeros([1]), trainable=self.__trainable, name="b2")
        }
        return biases
    
    def GetDimSRCNNOptimizer(self):
        optimizer = {
            "o1": tf.compat.v1.train.GradientDescentOptimizer(0.0001),
            "o2": tf.compat.v1.train.GradientDescentOptimizer(0.0001),
            "o3": tf.compat.v1.train.GradientDescentOptimizer(0.00001)
        }
        return optimizer

    def Get2DConv(self, idx, img, weights, biases, padding, strides=__normalStrides, activation=ReLU):
        conv = tf.nn.conv2d(input=img, filters=weights["w"+idx], strides=strides, padding=padding) + biases["b"+idx]
        if activation == self.ReLU:
            conv = tf.nn.relu(conv)
        return conv

    def __getLoss(self, labels, pred):
        return tf.reduce_mean(input_tensor=tf.square(labels - pred))    # loss函数为mse值

    def TrainSRCNN(self):
        self.__trainData, self.__trainLabel = self.DO.GetH5File(fileName=self.DO.TrainDataDir)
        weights = self.GetSRCNNWeights()
        biases = self.GetSRCNNBiases()
        optimizer = self.GetSRCNNOptimizer()
        conv1 = self.Get2DConv(idx="1", img=self.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding, activation=None)

        var_list1 = [weights["w1"], biases["b1"], weights["w2"], biases["b2"]]
        var_list2 = [weights["w3"], biases["b3"]]

        loss = self.__getLoss(self.Labels, conv3)
        grads = tf.gradients(ys=loss, xs=var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = optimizer["o1"].apply_gradients(zip(grads1, var_list1))
        train_op2 = optimizer["o2"].apply_gradients(zip(grads2, var_list2))
        train_op = tf.group(train_op1, train_op2)

        counter = 0
        start_time = time.time()
        saver=tf.compat.v1.train.Saver(max_to_keep=5)

        with tf.compat.v1.Session() as sess:
            print("Training...")
            sess.run(tf.compat.v1.initialize_all_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir="checkpoint-srcnn")
            if ckpt and ckpt.model_checkpoint_path:
                print("Continuing")
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            for ep in range(self.__epoch):
                # Run by batch images
                batch_idxs = len(self.__trainData) // self.__batchSize
                for idx in range(0, batch_idxs):
                    batch_images = self.__trainData[idx*self.__batchSize : (idx+1)*self.__batchSize]
                    batch_labels = self.__trainLabel[idx*self.__batchSize : (idx+1)*self.__batchSize]

                    counter +=1
                    _, err = sess.run([train_op, loss], feed_dict={self.Images: batch_images, self.Labels: batch_labels})

                    if counter % 10000 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-start_time, err))
                        saver.save(sess, os.path.join("checkpoint-srcnn", "SRCNN"), global_step=counter, write_meta_graph=False)

    def TrainDimSRCNN(self):
        self.__trainData, self.__trainLabel = self.DO.GetH5File(fileName=self.DO.TrainDataDir)
        weights = self.GetDimSRCNNWeights()
        biases = self.GetDimSRCNNBiases()
        optimizer = self.GetDimSRCNNOptimizer()
        conv1 = self.Get2DConv(idx="1", img=self.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding, activation=None)

        pred = conv3

        var_list1 = [weights["w1"], biases["b1"]]
        var_list2 = [weights["w2"], biases["b2"]]
        var_list3 = [weights["w3"], biases["b3"]]

        loss = self.__getLoss(self.Labels, pred)
        grads = tf.gradients(ys=loss, xs=var_list1+var_list2+var_list3)

        varLen = len(var_list1)
        grads1 = grads[0*varLen:1*varLen]
        grads2 = grads[1*varLen:2*varLen]
        grads3 = grads[2*varLen:3*varLen]

        train_op1 = optimizer["o1"].apply_gradients(zip(grads1, var_list1))
        train_op2 = optimizer["o2"].apply_gradients(zip(grads2, var_list2))
        train_op3 = optimizer["o3"].apply_gradients(zip(grads3, var_list3))

        train_op = tf.group(train_op1, train_op2, train_op3)

        counter = 0
        start_time = time.time() #记录开始时间
        saver = tf.compat.v1.train.Saver(max_to_keep=5)  #只保留最近的五个模型的参数值

        print("Training...")
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.initialize_all_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir="checkpoint-dim")
            if ckpt and ckpt.model_checkpoint_path:
                print("Continuing")
                saver.restore(sess, ckpt.model_checkpoint_path)
            for ep in range(self.__epoch):
                batch_idxs = len(self.__trainData) // self.__batchSize
                for idx in range(batch_idxs):
                    batch_images = self.__trainData[idx*self.__batchSize : (idx+1)*self.__batchSize]
                    batch_labels = self.__trainLabel[idx*self.__batchSize : (idx+1)*self.__batchSize]

                    counter +=1
                    _, err = sess.run([train_op, loss], feed_dict={self.Images: batch_images, self.Labels: batch_labels})

                    if counter % 50000 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-start_time, err))
                        saver.save(sess, os.path.join("checkpoint-dim", "SRCNN"), global_step=counter, write_meta_graph=False) 

if __name__ == "__main__":
    TR = Trainer(epoch=15000)
    # TR.TrainSRCNN()
    TR.TrainDimSRCNN()
