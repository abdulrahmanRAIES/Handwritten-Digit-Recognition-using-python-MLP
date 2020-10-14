from skimage.feature import hog


class hoggg:
    #def hogg(self,resizeImage):
        #return self.extractingg(resizeImage)

    def extractingg(self,resizeImage):
            fd, hog_image = hog (resizeImage,   orientations=9,  pixels_per_cell=(14, 14),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
            im_bw = list(hog_image.flatten())
            return im_bw
    