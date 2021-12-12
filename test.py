'''
Author: liziwei01
Date: 2021-12-12 02:33:21
LastEditors: liziwei01
LastEditTime: 2021-12-13 01:43:15
Description: file content
'''
import os
import h5py
import numpy as np
import tensorflow as tf
import time
import imageio
import scipy.ndimage
from prepare import DataOperations
from train import Trainer

class Tester:
    __trainable = True

    def __init__(self, do=DataOperations(), tr=Trainer(), padding="SAME") -> None:
        self.DO = do
        self.TR = tr
        self.__padding = padding
        tf.compat.v1.disable_eager_execution()

    def GetPSNR(self, lrImg, hrImg):
        (x,y)=np.shape(lrImg)
        img1=lrImg[3:(x-3),3:(y-3)] #减去边缘，matlab中即减去边缘的三个像素点
        img2=hrImg[3:(x-3),3:(y-3)] 
        diff = np.abs(img1*255.0- img2*255.0)
        mse = np.square(diff).mean() #mse表示当前图像与原有图像的均方误差
        psnr = 20 * np.log10(255 / np.sqrt(mse)) #评价指标即峰值信噪比
        return psnr

    def Test(self):
        self.__testData, self.__testLabel = self.DO.GetH5File(fileName=self.DO.TestDataDir)
        weights = self.TR.GetSRCNNWeights()
        biases = self.TR.GetSRCNNBiases()
        conv1 = self.TR.Get2DConv(idx="1", img=self.TR.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.TR.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.TR.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding, activation=None)

        saver=tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("checkpoint-srcnn")
            if ckpt and ckpt.model_checkpoint_path:  # 加载保存的模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # img1 = (weights['w1'].eval())  #查看卷积核是否在变化
                result = conv3.eval({self.TR.Images: self.__testData, self.TR.Labels: self.__testLabel}) # 得到训练后的结果
                result1 = result.squeeze()                                    # 降维
                result2 = np.around(result1, decimals=4)                       # 取小数点的后四位
                self.DO.SaveImage(result2, os.path.join(os.getcwd(), 'sample', "test_image.png"))                       
                label=self.__testLabel.squeeze()                              # label数据降维
                print(self.GetPSNR(label, result2))                           # 计算并打印pnsr值

    def TestDim(self):
        self.__testData, self.__testLabel = self.DO.GetH5File(fileName=self.DO.TestDataDir)

        weights = self.TR.GetDimSRCNNWeights()
        biases = self.TR.GetDimSRCNNBiases()
        conv1 = self.TR.Get2DConv(idx="1", img=self.TR.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.TR.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.TR.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding)
        conv4 = self.TR.Get2DConv(idx="4", img=conv3, weights=weights, biases=biases, padding=self.__padding)
        conv5 = self.TR.Get2DConv(idx="5", img=conv4, weights=weights, biases=biases, padding=self.__padding, activation=None)

        saver=tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("checkpoint")
            if ckpt and ckpt.model_checkpoint_path:  # 加载保存的模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # img1 = (weights['w1'].eval())  #查看卷积核是否在变化
                result = conv5.eval({self.TR.Images: self.__testData, self.TR.Labels: self.__testLabel}) # 得到训练后的结果
                result1 = result.squeeze()                                    # 降维
                result2 = np.around(result1, decimals=4)                       # 取小数点的后四位
                self.DO.SaveImage(result2, os.path.join(os.getcwd(), 'sample', "test_image.png"))                       
                label=self.__testLabel.squeeze()                              # label数据降维
                print(self.GetPSNR(label, result2))                           # 计算并打印pnsr值

if __name__ == "__main__":
    DO = DataOperations()
    TR = Trainer()
    TE = Tester(do=DO, tr=TR)
    TE.TestDim()