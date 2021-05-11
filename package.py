import cv2
import numpy as np 
import math
import pyautogui as p
import time as t



#Reading the  Front Camera
capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
def nothing(x):
    pass
#creating a  new window called Colour modifications
cv2.namedWindow("Colour modifications",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Colour modifications", (300, 300)) 
cv2.createTrackbar("Thresh", "Colour modifications", 0, 255, nothing)

cv2.createTrackbar("Lwr_h", "Colour modifications", 0, 255, nothing)
cv2.createTrackbar("Lwr_s", "Colour modifications", 0, 255, nothing)
cv2.createTrackbar("Lwr_v", "Colour modifications", 0, 255, nothing)
cv2.createTrackbar("Upr_h", "Colour modifications", 255, 255, nothing)
cv2.createTrackbar("Upr_s", "Colour modifications", 255, 255, nothing)
cv2.createTrackbar("Upr_v", "Colour modifications", 255, 255, nothing)


while True:
    _,frame = capture.read()
    frame = cv2.flip(frame,2)
    frame = cv2.resize(frame,(600,500))
    
    cv2.rectangle(frame, (0,1), (300,500), (255, 0, 0), 0)
    crp_image = frame[1:500, 0:300]
    
    
    hsv = cv2.cvtColor(crp_image, cv2.COLOR_BGR2HSV)
    #object(hand) detection
    l_h = cv2.getTrackbarPos("Lwr_h", "Colour modifications")
    l_s = cv2.getTrackbarPos("Lwr_s", "Colour modifications")
    l_v = cv2.getTrackbarPos("Lwr_v", "Colour modifications")

    u_h = cv2.getTrackbarPos("Upr_h", "Colour modifications")
    u_s = cv2.getTrackbarPos("Upr_s", "Colour modifications")
    u_v = cv2.getTrackbarPos("Upr_v", "Colour modifications")
    
    lwr_bd = np.array([l_h, l_s, l_v])
    upr_bd = np.array([u_h, u_s, u_v])
    
    
    mask = cv2.inRange(hsv, lwr_bd, upr_bd)
    
    filtr = cv2.bitwise_and(crp_image, crp_image, mask=mask)
    
    
    mask_1  = cv2.bitwise_not(mask)
    m_g = cv2.getTrackbarPos("Thresh", "Colour modifications") 
    ret,thresh = cv2.threshold(mask_1,m_g,255,cv2.THRESH_BINARY)
    dilation = cv2.dilate(thresh,(3,3),iterations = 6)
    
    #finding the countours
    
    cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    
    try:
        
        
        cm = max(cnts, key=lambda x: cv2.contourArea(x))
        
        ep = 0.0005*cv2.arcLength(cm,True)
        data= cv2.approxPolyDP(cm,ep,True)
    
        hull = cv2.convexHull(cm)
        
        cv2.drawContours(crp_image, [cm], -1, (50, 50, 150), 2)
        cv2.drawContours(crp_image, [hull], -1, (0, 255, 0), 2)
        
        hull = cv2.convexHull(cm, returnPoints=False)
        defects = cv2.convexityDefects(cm, hull)
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
           
            start = tuple(cm[s][0])
            end = tuple(cm[e][0])
            far = tuple(cm[f][0])
            #Rule of cosine
            a = math.sqrt((end[0] - start[0]) * 2 + (end[1] - start[1]) * 2)
            b = math.sqrt((far[0] - start[0]) * 2 + (far[1] - start[1]) * 2)
            c = math.sqrt((end[0] - far[0]) * 2 + (end[1] - far[1]) * 2)
            angle = (math.acos((b * 2 + c * 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            
            
            if angle <= 50:
                count_defects += 1
                cv2.circle(crp_image,far,5,[255,255,255],-1)
        
        print("count==",count_defects)
        
        
        if count_defects == 0:
            
            cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        elif count_defects == 1:
            
            p.press("space")
            cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 2:
            p.press("up")
            
            cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 3:
            p.press("down")
            
            cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 4:
            p.press("right")
            
            cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        else:
            pass
           
    except:
        pass
        
    cv2.imshow("Thresh", thresh)
  
    cv2.imshow("filter==",filtr)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(25) &0xFF    
    if key == ord('q'): 
        break
capture.release()
cv2.destroyAllWindows()
