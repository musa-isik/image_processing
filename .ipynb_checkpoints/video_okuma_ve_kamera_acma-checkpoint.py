import cv2
import time
import mediapipe as mp

def video_capture(bfr):
    capture = cv2.VideoCapture(0)
    
    while True:
        frame = capture.read()
        cv2.imshow("frame",frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): #q tuşu ile çıkış yapılacak
            break;
            
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
    
    bfr=0
    
    video_capture(bfr)