import cv2
import sys
import numpy
import ctypes
from PyQt5 import QtCore, QtGui, QtWidgets
from CV_hw1_ui import Ui_MainWindow
from scipy import signal



class MainWindow(QtWidgets.QMainWindow):
    path = sys.path[0] # get file path
    chihiroGaussian = sobelx = sobely = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Question 1
        self.ui.pushButton.clicked.connect(self.loadImg)
        self.ui.pushButton_2.clicked.connect(self.colorSeperation)
        self.ui.pushButton_3.clicked.connect(self.imageFlipping)
        self.ui.pushButton_4.clicked.connect(self.imgBlending)

        # Question 2
        self.ui.medium_filter_btn.clicked.connect(self.medianFilter)
        self.ui.gaussian_blur_btn.clicked.connect(self.gaussianBlur)
        self.ui.bilateral_filter.clicked.connect(self.bilateralFilter)

        # Question 3
        self.ui.gaussin_blur2_btn.clicked.connect(self.gaussianblur2)
        self.ui.sobelx_btn.clicked.connect(self.sobelX)
        self.ui.sobely_btn.clicked.connect(self.sobelY)
        self.ui.magnitude_btn.clicked.connect(self.magnitude)
        
        # Question 4
        self.ui.trans_btn.clicked.connect(self.transformation)

    # Quention 1 implement part

    # 1.1 Load Image
    def loadImg(self):
        #frame = QtWidgets.QFileDialog.getOpenFileName(self,'Open file','','Image files (*.jpg *.gif *.png *.jpeg)')
        '''
        frame is a tuple, which have two parameter, first one is path, second one is 'Image files (*.jpg *.gif *.png *.jpeg)'
        '''
        #img = cv2.imread(frame[0]) # frame[0] is the path of image

        img = cv2.imread("Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
        cv2.imshow("Image", img)
        print("Image Height : ",img.shape[0])
        print("Image Width : ",img.shape[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 1.2 Color Seperation
    def colorSeperation(self):
        img = cv2.imread("Dataset_opencvdl/Q1_Image/Flower.jpg")
        cv2.imshow("Flower", img)
        
        # b, g, r is single channel (灰階圖), notice that sequence is b, g, r
        b, g, r = cv2.split(img) 
        zeros = numpy.zeros(img.shape[:2], dtype = "uint8") # set zero matrix

        # do channel merge to 3 channel
        merged_b = cv2.merge([b,zeros,zeros])
        merged_g = cv2.merge([zeros,g,zeros])
        merged_r = cv2.merge([zeros,zeros,r])
        
        cv2.imshow("flower_blue", merged_b)
        cv2.imshow("flower_green",merged_g)
        cv2.imshow("flower_red",merged_r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 1.3 Image Flipping
    def imageFlipping(self):
        img = cv2.imread("Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
        ''' 
        second parameter of cv2.flip() is flipping mode
        0 for vertical flipping, 1 for horizontal flipping,
        -1 for vertical and horizontal flipping
        '''
        img_flip = cv2.flip(img, 1)
        cv2.imshow("img_origin", img)
        cv2.imshow("img_flipping", img_flip)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 1.4 Blending
    def blending(self, val):
        alpha = val / 255
        beta = ( 1.0 - alpha )
        img = cv2.addWeighted(self.img1, alpha, self.img2, beta, 0.0)
        cv2.imshow("img", img)

    def imgBlending(self):
        self.img1 = cv2.imread("Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
        self.img2 = cv2.flip(self.img1, 1)
        cv2.imshow("img", self.img1)
        cv2.createTrackbar('tracker','img', 0, 255, self.blending)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Question 2 Implement part

    # 2.1 Median Filter
    def medianFilter(self):
        imgCat = cv2.imread("Dataset_opencvdl/Q2_Image/Cat.png")
        # cv2.imshow("Cat",imgCat)
        '''
            medianBlur() : first parameter is source, 
            # second parameter is the size of box filter, it must be positive and odd
        '''
        median = cv2.medianBlur(imgCat, 7)
        compare = numpy.concatenate((imgCat, median), axis = 0)
        cv2.imshow("Median Comparison", compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2.2 Gaussian Blur
    def gaussianBlur(self):
        imgCat2 = cv2.imread("Dataset_opencvdl/Q2_Image/Cat.png")
        '''
            GaussianBlur():the first parameter is source
            second parameter is width and height of kernel, which should be positive and odd
            third parameter is sigmaX and sigmaY,  If only sigmaX is specified, sigmaY is taken as equal to sigmaX. 
            If both are given as zeros, they are calculated from the kernel size.
        '''
        gaussian = cv2.GaussianBlur(imgCat2, (3,3), 0)
        compare3 = numpy.concatenate((imgCat2, gaussian), axis = 0)
        cv2.imshow("Gaussin Comparison", compare3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 2.3 Bilateral Filter
    def bilateralFilter(self):
        imgCat3 = cv2.imread("Dataset_opencvdl/Q2_Image/Cat.png")
        '''
            bilateralfilter():four arguments, second is Diameter of each pixel neighborhood
            third parameter is sigmaColor, The greater the value, the colors farther to each other will start to get mixed.
            forth parameter is sigmaSpace, The greater its value, the more further pixels will mix together, 
                given that their colors lie within the sigmaColor range.
        '''
        bilateral = cv2.bilateralFilter(imgCat3, 9, 90, 90)
        compare4 = numpy.concatenate((imgCat3, bilateral), axis = 0)
        cv2.imshow("Bilateral Comparison", compare4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 3 Implement part

    # the following three funcs will setting the global variable chihiroGaussian, sobelx and sobely 
    def Gaussian_func(self):
        self.chihiroGrayscale = cv2.imread("Dataset_opencvdl/Q3_Image/Chihiro.jpg", 0)
        # Generate Gaussian filter
        x,y = numpy.mgrid[-1:2, -1:2 ]
        gaussian_kernel = numpy.exp(-(x**2 + y**2))
        # Normalization filter
        gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
        self.chihiroGaussian = cv2.filter2D(self.chihiroGrayscale, -1, gaussian_kernel)

    def Sobel_X(self):
        self.Gaussian_func()
        self.sobelXfilter = numpy.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=numpy.float)
        self.sobelx = signal.convolve2d(self.chihiroGaussian, self.sobelXfilter, "same", "symm")
        self.sobelx = numpy.uint8(numpy.absolute(self.sobelx))

    def Sobel_Y(self):
        self.Gaussian_func()
        self.sobelYfilter = numpy.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype = numpy.float)
        self.sobely = cv2.filter2D(self.chihiroGaussian, -1, self.sobelYfilter)
        self.sobely = numpy.uint8(numpy.absolute(self.sobely))

    # 3.1 Gaussian Blur
    def gaussianblur2(self):
        self.Gaussian_func()
        cv2.imshow("Chihiro Gaussian", self.chihiroGaussian)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 3.2 Sobel X, vertical

    def sobelX(self):
        self.Sobel_X()
        cv2.imshow("Chihiro SobelX", self.sobelx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 3.3 Sobel Y, horizontal
    def sobelY(self):
        self.Sobel_Y()
        cv2.imshow("Chihiro SobelY", self.sobely)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 3.4 Magnitude
    def magnitude(self):
        self.Gaussian_func()
        self.Sobel_X()
        self.Sobel_Y()
        Chihiro_magnitude = cv2.bitwise_or(self.sobelx, self.sobely)
        cv2.imshow("Chihiro Magnitude", Chihiro_magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 4 Implement part

    # 4. Transformation
    def transformation(self):
        parrot = cv2.imread("Dataset_opencvdl/Q4_Image/Parrot.png")

        rotation = self.ui.edit_retation.text()
        scaling = self.ui.edit_scaling.text()
        tx = self.ui.edit_tx.text()
        ty = self.ui.edit_ty.text()

        if rotation == "" or scaling == "" or tx == "" or ty == "":
            ctypes.windll.user32.MessageBoxW(0, "Fill out the form, OK ?", "可憐吶", 1)
            return
        if not rotation.isdigit() or not scaling.isdigit() or not tx.isdigit() or not ty.isdigit():
            ctypes.windll.user32.MessageBoxW(0, "Input should only be digits and positive integer, OK ?", "可憐吶", 1)
            return

        rows = parrot.shape[0]
        cols = parrot.shape[1]
        M = numpy.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
        # it should trasform before rotation, cause if it rotate first, the image may be cutted down
        parrot_trans = cv2.warpAffine(parrot, M, (cols, rows))

        new_x = 160 + int(tx)
        new_y = 160 + int(ty)

        H = cv2.getRotationMatrix2D((new_x, new_y), numpy.float32(rotation), numpy.float32(scaling))
        parrot_res = cv2.warpAffine(parrot_trans, H, (cols, rows))
        cv2.imshow("Parrot Result", parrot_res)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())