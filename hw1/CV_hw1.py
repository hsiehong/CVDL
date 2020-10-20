import cv2
import sys
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from CV_hw1_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    path = sys.path[0] # get file path
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # question 1
        self.ui.pushButton.clicked.connect(self.button1Clicked)
        self.ui.pushButton_2.clicked.connect(self.button2Clicked)
        self.ui.pushButton_3.clicked.connect(self.button3Clicked)
        self.ui.pushButton_4.clicked.connect(self.button4Clicked)

    def button1Clicked(self):
        frame = QtWidgets.QFileDialog.getOpenFileName(self,'Open file','','Image files (*.jpg *.gif *.png *.jpeg)')
        # frame is a tuple, which have two parameter, first one is path, second one is 'Image files (*.jpg *.gif *.png *.jpeg)'
  
        img = cv2.imread(frame[0]) # frame[0] is the path of image
        cv2.imshow("Image", img)
        print("Image Height : ",img.shape[0])
        print("Image Width : ",img.shape[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def button2Clicked(self):
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
        
    def button3Clicked(self):
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
    
    def blending(self, val):
        alpha = val / 255
        beta = ( 1.0 - alpha )
        img = cv2.addWeighted(self.img1, alpha, self.img2, beta, 0.0)
        cv2.imshow("img", img) # this line will modified origin pic because the file name is same

    def button4Clicked(self):
        self.img1 = cv2.imread(self.path+"/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
        self.img2 = cv2.flip(self.img1, 1)
        cv2.imshow("img", self.img1)
        cv2.createTrackbar('tracker','img', 0, 255, self.blending)
    

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())