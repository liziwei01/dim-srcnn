'''
Author: liziwei01
Date: 2021-12-12 02:33:21
LastEditors: liziwei01
LastEditTime: 2022-01-02 11:17:04
Description: file content
'''
import os
import h5py
import numpy as np
from six import reraise
import tensorflow as tf
import time
import imageio
import scipy.ndimage
from prepare import DataOperations
from train import Trainer
import time
from PIL import Image
import skimage.metrics

class DSRCNNImgName:
    def __init__(self, imgAbsPath, upScale, resizeMethodName) -> None:
        self.ImgAbsPath = imgAbsPath
        self.ImgName = self.__getImgName(imgAbsPath)
        self.UpscaleFactor = upScale
        self.ResizeMethodName = resizeMethodName

    def __getImgName(self, imgAbsPath):
        cwd = os.getcwd()
        imgRelPath = imgAbsPath.replace(cwd, "").replace("/Test/", "").replace(".jpeg", "")
        return imgRelPath

    def GetSavePath(self, preName="", postName=""):
        fileName = preName + self.string() + postName + ".png"
        savedImgName = os.path.join(os.getcwd(), 'sample', fileName)
        return savedImgName

    def string(self):
        return self.ImgName + "Upscale"+ str(self.UpscaleFactor) + "xBy" + self.ResizeMethodName

    def print(self):
        print(self.string() + ": ")

class DSRCNNResult:
    dsrcnn = 0
    nn = 1
    bilinear = 2
    bicubic = 3
    time = 4
    psnr = 5
    ssim = 6

    def __init__(self) -> None:
        self.__result = {
            self.dsrcnn: {},
            self.nn: [],
            self.bilinear: [],
            self.bicubic: [],
        }

    def __getSampleImgInfoDict(self):
        return {
            self.time: []

        }


    def recordDSRCNN(self):
        self.__result[self.dsrcnn] = ""

class Tester:
    __trainable = True
    __resizeMax = 5
    __resizeMin = 4

    def __init__(self, do=DataOperations(), tr=Trainer(), padding="SAME", lrScale=3) -> None:
        self.DO = do
        self.TR = tr
        self.__padding = padding
        self.__lrScale = lrScale
        tf.compat.v1.disable_eager_execution()

    def GetPSNR(self, hrImg, lrImg):
        # (x, y) = np.shape(lrImg)
        # img1 = lrImg[3:(x-3),3:(y-3)] # get rid of margin
        # img2 = hrImg[3:(x-3),3:(y-3)]
        # # diff = np.abs(img1 - img2)
        # diff = np.abs(img1*255.0 - img2*255.0)
        # mse = np.square(diff).mean()
        # psnr = 20 * np.log10(255 / np.sqrt(mse))
        # return psnr
        return skimage.metrics.peak_signal_noise_ratio(lrImg, hrImg)

    def GetSSIM(self, lrImg, hrImg):
        return skimage.metrics.structural_similarity(lrImg, hrImg)

    def Normalize(self, v):
        norm = np.linalg.norm(v)
        if norm < 0.00000001: 
            return v
        return v / norm

    def __dropUselessInfo(self, imgArr):
        max = np.max(imgArr)
        if max > 1:
            print("bigger than 1!!!: ", str(max))
            diff = max - 1
            imgArr = imgArr - diff
        return np.around(np.float64(imgArr.squeeze()), decimals=4)

    def __imageArrNormalize(self, img):
        img = self.Normalize(np.float64(img.squeeze()))
        return np.around(img, decimals=4)

    def __getGroundTruthImg(self, imgY):
        if self.__lrScale <= 1:
            return imgY
        # downscale the resolution by lrScale
        w, h = imgY.size
        groundTruthImg = imgY.resize((w//self.__lrScale, h//self.__lrScale), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        return groundTruthImg

    def __getResizeDict(self):
        return {
            "NEAREST": Image.NEAREST, 
            "BILINEAR": Image.BILINEAR, 
            "BICUBIC": Image.BICUBIC, 
        }

    def __test(self):
        self.__testData, self.__testLabel = self.DO.GetH5File(fileName=self.DO.TestDataDir)
        weights = self.TR.GetSRCNNWeights()
        biases = self.TR.GetSRCNNBiases()
        conv1 = self.TR.Get2DConv(idx="1", img=self.TR.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.TR.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.TR.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding, activation=None)

        pred = conv3

        saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("checkpoint-srcnn")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                startTime = time.time()
                result = pred.eval({self.TR.Images: self.__testData, self.TR.Labels: self.__testLabel})
                endTime = time.time()

                result = np.asarray(np.around(result.squeeze(), decimals=4), dtype = np.float64)
                label = self.__testLabel.squeeze()

                self.DO.SaveImage(result, os.path.join(os.getcwd(), 'sample', "test_srcnn.png"))

                print("srcnnImage:")
                print("time: [%f]" % (endTime-startTime))
                print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(label, result), self.GetSSIM(label, result)))

    def Test(self):
        img = Image.open("./Test/VCG21002a4d860.jpeg")
        self.DO.SaveImage(img, os.path.join(os.getcwd(), 'sample', "ori_colored.png"))
        # img = img.convert("RGB")
        # self.DO.SaveImage(img, os.path.join(os.getcwd(), 'sample', "RGB_colored.png"))
        img = img.convert("YCbCr")
        self.DO.SaveImage(img, os.path.join(os.getcwd(), 'sample', "YCbCr_colored.png"))
        imgY, imgCb, imgCr = img.split()
        # imgArr = np.array(imgY) / 255.0
        w, h = imgY.size
        midImg = imgY.resize(( w//self.__lrScale, h//self.__lrScale), Image.BICUBIC).resize(( w, h), Image.BICUBIC)
        
        imgArr = np.array(midImg) / 255.0

        midImg = midImg.resize(( w//3, h//3), Image.BICUBIC)
        bicubicImage = midImg.resize((w, h), Image.BICUBIC)

        # self.__testData, self.__testLabel = self.DO.GetH5File(fileName=self.DO.TestDataDir)
        # h, w = self.DO.GetImageHeightWidth(imgArr)
        imgArr = imgArr.reshape([1, h, w, 1])
        bicubicImageArr = np.array(bicubicImage).reshape([1, h, w, 1]) / 255.0

        weights = self.TR.GetSRCNNWeights()
        biases = self.TR.GetSRCNNBiases()
        conv1 = self.TR.Get2DConv(idx="1", img=self.TR.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.TR.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.TR.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding, activation=None)

        pred = conv3

        saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("checkpoint-srcnn")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                startTime = time.time()
                result = pred.eval({self.TR.Images: bicubicImageArr, self.TR.Labels: imgArr})
                endTime = time.time()

                result = self.__dropUselessInfo(result)
                label = self.__dropUselessInfo(imgArr)

                self.DO.SaveImage(result*255, os.path.join(os.getcwd(), 'sample', "test_srcnn.png"))

                YColoredResult = Image.fromarray(result*255).convert("L")
                coloredResult = Image.merge("YCbCr", (YColoredResult, imgCb, imgCr)).convert("RGB")
                self.DO.SaveImage(coloredResult, os.path.join(os.getcwd(), 'sample', "test_srcnn_colored.png"))
                coloredResult = Image.merge("YCbCr", (imgY, imgCb, imgCr)).convert("RGB")
                self.DO.SaveImage(coloredResult, os.path.join(os.getcwd(), 'sample', "srcnn_colored.png"))

                print("dimImage:")
                print("time: [%f]" % (endTime-startTime))
                print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(label, result), self.GetSSIM(label, result)))


    def TestDim(self):
        # get D-SRCNN model
        weights = self.TR.GetDimSRCNNWeights()
        biases = self.TR.GetDimSRCNNBiases()
        conv1 = self.TR.Get2DConv(idx="1", img=self.TR.Images, weights=weights, biases=biases, padding=self.__padding)
        conv2 = self.TR.Get2DConv(idx="2", img=conv1, weights=weights, biases=biases, padding=self.__padding)
        conv3 = self.TR.Get2DConv(idx="3", img=conv2, weights=weights, biases=biases, padding=self.__padding, activation=None)
        pred = conv3
        saver = tf.compat.v1.train.Saver()
        
        imagesPaths = self.DO.GetImagesPaths(self.DO.TestDataDir)
        for imgAbsPath in imagesPaths:
            # read test img and convert to YCbCr
            img = Image.open(imgAbsPath).convert("YCbCr")

            # keep Y channel only
            imgY, imgCb, imgCr = img.split()
            w, h = imgY.size

            # get groundtruth
            groundTruthImg = self.__getGroundTruthImg(imgY)

            # prepare the inputs for D-SRCNN
            # # first the groundtruth
            groundTruthImgArrForCompare = (np.array(groundTruthImg) / 255.0)
            groundTruthImgArr = groundTruthImgArrForCompare.reshape([1, h, w, 1])
            # # then the inputs
            DSRCNNInputs = {}
            DSRCNNMidInputs = {}
            resizeDict = self.__getResizeDict()
            for upScale in range(self.__resizeMin, self.__resizeMax):
                for resizeMethodName, resizeMethod in resizeDict.items():
                    midImg = groundTruthImg.resize(( w//upScale, h//upScale), resizeMethod)
                    inputImg = midImg.resize((w, h), resizeMethod)
                    inputImgArr = (np.array(inputImg) / 255.0).reshape([1, h, w, 1])
                    imgName = DSRCNNImgName(imgAbsPath, upScale, resizeMethodName)
                    DSRCNNInputs[imgName] = inputImgArr
                    # ignore this
                    DSRCNNMidInputs[imgName] = midImg

            # start predict hr images
            with tf.compat.v1.Session() as sess:
                ckpt = tf.train.get_checkpoint_state("checkpoint-dim")
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("prediction:")
                    for imgName, inputImgArr in DSRCNNInputs.items():
                        # predict hr image
                        startTime = time.time()
                        predictHrImgArr = pred.eval({self.TR.Images: inputImgArr, self.TR.Labels: groundTruthImgArr})
                        endTime = time.time()

                        predictHrImgArr = self.__dropUselessInfo(predictHrImgArr)

                        # visualize model output
                        predictHrImg = Image.fromarray(predictHrImgArr*255).convert("L")
                        predictHrImgColor = Image.merge("YCbCr", (predictHrImg, imgCb, imgCr)).convert("RGB")
                        self.DO.SaveImage(predictHrImgColor, imgName.GetSavePath(preName="DSRCNN_"))
                        imgName.print()
                        print("N-SRCNN: ")
                        print("time: [%f]" % (endTime-startTime))
                        print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(groundTruthImgArrForCompare, predictHrImgArr), self.GetSSIM(groundTruthImgArrForCompare, predictHrImgArr)))

                        # compare with usual SR method
                        # for resizeMethodName, resizeMethod in resizeDict.items():
                        #     midImg = DSRCNNMidInputs[imgName]
                        #     startTime = time.time()
                        #     predictHrImg = midImg.resize((w, h), resizeMethod)
                        #     endTime = time.time()

                        #     predictHrImgColor = Image.merge("YCbCr", (predictHrImg.convert("L"), imgCb, imgCr)).convert("RGB")
                        #     self.DO.SaveImage(predictHrImgColor, imgName.GetSavePath(preName=resizeMethodName + ":"))

                        #     bilinearImgArr = (np.array(predictHrImg) / 255.0)
                        #     print(resizeMethodName + ":")
                        #     print("time: [%f]" % (endTime-startTime))
                        #     print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(groundTruthImgArrForCompare, bilinearImgArr), self.GetSSIM(groundTruthImgArrForCompare, bilinearImgArr)))

                        

    def CompareOtherMethods(self):
        img = Image.open("./Test/VCG21002a4d860.jpeg").convert("YCbCr")
        img, _, _ = img.split()
        w, h = img.size

        groundTruth = img.resize(( w//self.__lrScale, h//self.__lrScale), Image.BICUBIC).resize(( w, h), Image.BICUBIC)
        imgArr = np.array(groundTruth)

        midImg = groundTruth.resize(( w//self.DO.Scale, h//self.DO.Scale), Image.BICUBIC)
        
        nearestStartTime = time.time()
        nearestImage = midImg.resize((w, h), Image.NEAREST)

        bilinearStartTime = time.time()
        bilinearImage = midImg.resize((w, h), Image.BILINEAR)

        bicubicStartTime = time.time()
        bicubicImage = midImg.resize((w, h), Image.BICUBIC)
        bicubicEndTime = time.time()

        nearestImage.save("./sample/nearest_image.png")
        bilinearImage.save("./sample/bilinear_image.png")
        bicubicImage.save("./sample/bicubic_image.png")

        print("nearestImage:")
        print("time: [%f]" % (bilinearStartTime-nearestStartTime))
        print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(imgArr, np.array(nearestImage)), self.GetSSIM(imgArr, np.array(nearestImage))))

        print("bilinearImage:")
        print("time: [%f]" % (bicubicStartTime-bilinearStartTime))
        print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(imgArr, np.array(bilinearImage)), self.GetSSIM(imgArr, np.array(bilinearImage))))

        print("bicubicImage:")
        print("time: [%f]" % (bicubicEndTime-bicubicStartTime))
        print("psnr: [%f], ssim: [%f]" % (self.GetPSNR(imgArr, np.array(bicubicImage)), self.GetSSIM(imgArr, np.array(bicubicImage))))

if __name__ == "__main__":
    TE = Tester(lrScale=1)
    # TE.Test()
    # TE.CompareOtherMethods()
    TE.TestDim()
