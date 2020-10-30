import cv2
import sys
import signal
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from CV_hw1_ui2 import Ui_MainWindow



class MainWindow(QtWidgets.QMainWindow):
    path = sys.path[0] # get file path
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

        # Question 4

    # Quention 1 implement part

    # 1.1 Load Image
    def loadImg(self):
        frame = QtWidgets.QFileDialog.getOpenFileName(self,'Open file','','Image files (*.jpg *.gif *.png *.jpeg)')
        # frame is a tuple, which have two parameter, first one is path, second one is 'Image files (*.jpg *.gif *.png *.jpeg)'
  
        img = cv2.imread(frame[0]) # frame[0] is the path of image
        cv2.imshow("Image", img)
        print("Image Height : ",img.shape[0])
        print("Image Width : ",img.shape[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 1.2 Color Seperation
    def colorSeperation(self):
        img = cv2.imread(self.path+"/Dataset_opencvdl/Q1_Image/Flower.jpg")
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
        img = cv2.imread(self.path+"/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
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
        cv2.imshow("img", img) # this line will modified origin pic because the file name is same

    def imgBlending(self):
        self.img1 = cv2.imread(self.path+"/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
        self.img2 = cv2.flip(self.img1, 1)
        cv2.imshow("img", self.img1)
        cv2.createTrackbar('tracker','img', 0, 255, self.blending)
    
    # Question 2 Implement part

    # 2.1 Median Filter
    def medianFilter(self):
        imgCat = cv2.imread(self.path+"/Dataset_opencvdl/Q2_Image/Cat.png")
        # cv2.imshow("Cat",imgCat)
        '''
            medianBlur() : first parameter is source, 
            # second parameter is the size of box filter, it must be positive and odd
        '''
        median = cv2.medianBlur(imgCat, 7)
        compare = numpy.concatenate((imgCat, median), axis = 0)
        cv2.imshow("Median_Comparison", compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2.2 Gaussian Blur
    def gaussianBlur(self):
        imgCat2 = cv2.imread(self.path+"/Dataset_opencvdl/Q2_Image/Cat.png")
        '''
            GaussianBlur():the first parameter is source
            second parameter is width and height of kernel, which should be positive and odd
            third parameter is sigmaX and sigmaY,  If only sigmaX is specified, sigmaY is taken as equal to sigmaX. 
            If both are given as zeros, they are calculated from the kernel size.
        '''
        gaussian = cv2.GaussianBlur(imgCat2, (3,3), 0)
        compare3 = numpy.concatenate((imgCat2, gaussian), axis = 0)
        cv2.imshow("Gaussin_comparison", compare3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 2.3 Bilateral Filter
    def bilateralFilter(self):
        imgCat3 = cv2.imread(self.path+"/Dataset_opencvdl/Q2_Image/Cat.png")
        '''
            bilateralfilter():four arguments, second is Diameter of each pixel neighborhood
            third parameter is sigmaColor, The greater the value, the colors farther to each other will start to get mixed.
            forth parameter is sigmaSpace, The greater its value, the more further pixels will mix together, 
                given that their colors lie within the sigmaColor range.
        '''
        bilateral = cv2.bilateralFilter(imgCat3, 9, 90, 90)
        compare4 = numpy.concatenate((imgCat3, bilateral), axis = 0)
        cv2.imshow("Bilateral_comparison", compare4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 3 Implement part

    # 3.1 Gaussian Blur
    def gaussianblur2(self):
        self.chihiro = cv2.imread(self.path+"/Dataset_opencvdl/Q3_Image/Chihiro.jpg")
        cv2.imshow("Chihiro", self.chihiro)
        #self.chihiroGrayscale = cv2.cvtColor(self.chihiro, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("ChihiroGray", self.chihiroGrayscale)
        #self.combine = numpy.concatenate((self.chihiro, self.chihiroGrayscale), axis = 1)
        #cv2.imshow("Chihiro result", self.combine)
        
        # Generate Gaussian filter
        x,y = numpy.mgrid[-1:2, -1:2 ]
        gaussian_kernel = numpy.exp(-(x**2+y**2))
        
        # Normalization
        gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()

        # grad = signal.convolve2d(self.chihiroGrayscale, gaussian_kernel, boundary = 'symm', mode = 'same')
        # cv2.imshow("grad", grad)

        G_init = [[(-1, -1), (0, -1), (1, -1)],
                [(-1, 0), (0, 0), (1, 0)],
                [(-1, 1), (0, 1), (1, 1)]
                ]

        #cv2dfilter
        #normalixe


    # 3.2 Sobel X
    def sobelX(self):
        return

    # 3.3 Sobel Y
    def sobelY(self):
        return
    # 3.4 Magnitude
    def magnitude(self):
        return 

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())