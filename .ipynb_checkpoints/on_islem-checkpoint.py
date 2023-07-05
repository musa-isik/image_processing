import cv2

#görüntü boyutlandırma
def boyutlandirma(img,x,y):
    return cv2.resize(img,(x,y))