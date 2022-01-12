'''
Author: liziwei01
Date: 2021-12-11 21:32:42
LastEditors: liziwei01
LastEditTime: 2022-01-13 04:53:26
Description: file content
'''
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from imageio import imread
from PIL import Image
import imageio
import numpy as np
import scipy.misc as smi
import tensorflow as tf

class DataOperations:
    TrainDataDir = "Train"
    TestDataDir = "Test"

    def __init__(self, isTrain=True, scale=3, imageSize=33, stride=14, labelSize=21):
        self.__isTrain = isTrain
        self.Scale = scale
        self.__imageSize = imageSize
        self.__stride = stride
        self.__labelSize = labelSize
        self.__padding = abs(self.__imageSize - self.__labelSize) / 2 # 6像素点的边缘
        
    def __rgb2ycbcr(self, img, only_y=True):
        '''
        对应matlab的rgb2ycbcr函数
        提取Y通道
        same as matlab rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)
    
    def __readYChannel(self, path):
        return self.__rgb2ycbcr(imread(path))

    def SaveImage(self, image, path):
        return imageio.imsave(path, image)

    def __getFullPath(self, fileName):
        return os.path.join(os.getcwd(), fileName)
    
    def GetImagesPaths(self, dataset=TrainDataDir):
        dataDir = self.__getFullPath(dataset)
        data = glob.glob(os.path.join(dataDir, "*.jpeg"))
        return data

    # def __getFolderContainedFileNames(self, dataset):
    #     return os.listdir(dataset)

    def GetImageHeightWidth(self, imgArr):
        if len(imgArr.shape) == 3:
            h, w, _ = imgArr.shape
        else:
            h, w = imgArr.shape
        return h, w
    
    def GetScaleDivisibleHeightWidth(self, imgArr):
        h, w = self.GetImageHeightWidth(imgArr)
        h = h - np.mod(h, self.Scale)
        w = w - np.mod(w, self.Scale)
        return h, w

    def __cutImageToScaleDivisible(self, imgArr):
        h, w = self.GetScaleDivisibleHeightWidth(imgArr)
        if len(imgArr.shape) == 3:
            cuttedImage = imgArr[0:h, 0:w, :]
        else:
            cuttedImage = imgArr[0:h, 0:w]
        return cuttedImage

    def __getCuttedImage(self,img):
        cuttedImage = self.__cutImageToScaleDivisible(img)
        return cuttedImage / 255.

    def __getScaleDivisibleImage(self, img):
        cuttedImage = self.__getCuttedImage(img)
        return Image.fromarray(cuttedImage)

    # np to Image
    def GetSmallSizeImage(self, imgArr):
        h, w = self.GetScaleDivisibleHeightWidth(imgArr)
        cuttedImage = self.__getScaleDivisibleImage(imgArr)
        return cuttedImage.resize(( w//self.Scale, h//self.Scale), Image.BICUBIC)

    def __getLRImage(self, img):
        h, w = self.GetScaleDivisibleHeightWidth(img)
        midImg = self.GetSmallSizeImage(img)
        lrImg = midImg.resize((w, h), Image.BICUBIC)
        return np.around(np.float64(lrImg), decimals=4)

    def __getTargetCuttedImage(self, img):
        return np.around(self.__getCuttedImage(img), decimals=4)

    def __saveH5(self, arrData, arrLabel, fileName):
        savePath = os.path.join(os.getcwd(), "h5", fileName.lower()+".h5")
        with h5py.File(savePath, 'w') as hf:
            hf.create_dataset('data', data=arrData)
            hf.create_dataset('label', data=arrLabel)

    def __saveAsPreparedH5(self, subInputSequence, subLabelSequence, fileName):
        arrData = np.asarray(subInputSequence) # [?, 33, 33, 1]
        arrLabel = np.asarray(subLabelSequence) # [?, 21, 21, 1]
        self.__saveH5(arrData=arrData, arrLabel=arrLabel, fileName=fileName)

    def GetH5File(self, fileName="train"):
        dataDir = os.path.join(os.getcwd(), "h5", fileName.lower()+".h5")
        with h5py.File(dataDir, "r") as hf:
            trainData = np.array(hf.get("data"))
            trainLabel = np.array(hf.get("label"))
            return trainData, trainLabel

    def __saveSampleImage(self, img, fileName):
        imagePath = os.path.join(os.getcwd(), "sample", fileName.lower()+".png")
        self.SaveImage(img, imagePath)

    def PrepareTrainingData(self):
        subInputSequence = []
        subLabelSequence = []
        imagesPaths = self.GetImagesPaths(self.TrainDataDir)
        # 遍历每张图片用于训练
        for i in range(len(imagesPaths)):
            imgArr = self.__readYChannel(imagesPaths[i])
            label_= self.__getTargetCuttedImage(imgArr)  # 原图需要切割以 fit scale
            input_ = self.__getLRImage(imgArr)           # 不光切割同时也降低了原图的分辨率
            h, w = self.GetImageHeightWidth(input_)
            # 以self.Stride为步长进行取子图片操作
            for x in range(0, h-self.__imageSize+1, self.__stride):  
                for y in range(0, w-self.__imageSize+1, self.__stride):
                    sub_input = input_[x:x+self.__imageSize, y:y+self.__imageSize] # [33 x 33]
                    sub_label = label_[x+int(self.__padding):x+int(self.__padding)+self.__labelSize, y+int(self.__padding):y+int(self.__padding)+self.__labelSize] # [21 x 21]
            
                    # Make channel value
                    sub_input = sub_input.reshape([self.__imageSize, self.__imageSize, 1])  
                    sub_label = sub_label.reshape([self.__labelSize, self.__labelSize, 1])
            
                    subInputSequence.append(sub_input) # append为在列表末尾添加新的对象
                    subLabelSequence.append(sub_label)
        self.__saveAsPreparedH5(subInputSequence, subLabelSequence, self.TrainDataDir)

    def PrepareTestData(self):
        subInputSequence = []
        subLabelSequence = []
        imagesPaths = self.GetImagesPaths(self.TestDataDir)
        for i in range(len(imagesPaths)):
            image = self.__readYChannel(imagesPaths[i])
            label_= self.__getTargetCuttedImage(image)  # 原图需要切割以 fit scale
            input_ = self.__getLRImage(image)           # 不光切割同时也降低了原图的分辨率
            h, w = self.GetImageHeightWidth(input_)
            
            self.__saveSampleImage(label_, "label_image")   #保存真图
            self.__saveSampleImage(input_, "input_image")   #保存输入图片
            
            sub_input = input_.reshape([h, w, 1])  
            sub_label = label_.reshape([h, w, 1])
        
            subInputSequence.append(sub_input)
            subLabelSequence.append(sub_label)

        self.__saveAsPreparedH5(subInputSequence, subLabelSequence, self.TestDataDir)

def down():
    DO = DataOperations()
    imagesPaths = DO.GetImagesPaths(DO.TestDataDir)
    for i in range(len(imagesPaths)):
        img = Image.open(imagesPaths[i])
        w, h = img.size
        midImg = img.resize(( w//10, h//10), Image.BICUBIC)
        # midImg.save(imagesPaths[i])
        bicubicImage = midImg.resize((w, h), Image.BICUBIC)
        bicubicImage.save(imagesPaths[i])


if __name__ == "__main__":
    # DO = DataOperations()
    # DO.PrepareTrainingData()
    # DO.PrepareTestData()
    # down()
    DIMDO = DataOperations(imageSize=33, labelSize=27, stride=14)
    DIMDO.PrepareTrainingData()
    DIMDO.PrepareTestData()
    

