import os
import cv2
from skimage.feature import hog
from HistogramofOrientedGradient  import hoggg

class extract:
    def extracting(self):
        path = 'myData'  #name of file that contain the images
    #### IMPORTING DATA/IMAGES FROM FOLDERS
        images = []     # LIST CONTAINING ALL THE IMAGES
        classNo = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
        myList = os.listdir(path) #LIST THE NUMBER OF CLASSES 
        print("Total Classes Detected:",len(myList))
        noOfClasses = len(myList)
        print("Importing Classes .......")
        for x in range (0,noOfClasses):
            myPicList = os.listdir(path+"/"+str(x))
            for y in myPicList:
                curImg = cv2.imread(path+"/"+str(x)+"/"+y,cv2.IMREAD_GRAYSCALE)  
                curImg = cv2.medianBlur(curImg, 5)
                retval,mask_img = cv2.threshold(curImg,160, 255, cv2.THRESH_BINARY)
                #curImg = cv2.imread(path+"/"+str(x)+"/"+y)
                mask_img = cv2.resize(mask_img,(28,28), interpolation=cv2.INTER_AREA)
                images.append(hoggg().extractingg(mask_img))
                #images.append(hogg(mask_img))
                classNo.append(x)
            print(x,end= " ")
        print(" ")
        print("Total Images in Images List = ",len(images))
        print("Total IDS in classNo List= ",len(classNo))
    
        return images,classNo












    #return 0