import cv2
import urllib.request
import numpy as np

#if your running this code you need to have the numpy and opencv libraries installed 

def nothing(x):
    pass


url = 'http://192.168.100.142/800x600.jpg' #Paste the url of the cam your using this code is built around collecting data from a website but we can try having the camera send data straight to this later to optimize

# Setup Windows
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)  
cv2.createTrackbar("LS", "Tracking", 50, 255, nothing) 
cv2.createTrackbar("LV", "Tracking", 100, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 30, 179, nothing)  
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)     #we can adjust the colors from the code cuz the sliders lowk dont work lmao

while True:
    try:

        img_resp = urllib.request.urlopen(url, timeout=2)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        
        if frame is None:
            continue


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 4. Get Trackbar positions
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        # 5. Thresholding
        lower_b = np.array([l_h, l_s, l_v])
        upper_b = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_b, upper_b)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            if radius > 10:
          
                cx, cy = int(x), int(y)
                length = 20 

                cv2.line(frame, (cx - length, cy), (cx + length, cy), (0, 255, 0), 2)
                # Vertical line
                cv2.line(frame, (cx, cy - length), (cx, cy + length), (0, 255, 0), 2)
                
                cv2.putText(frame, f"Pos: {cx},{cy}", (cx + 15, cy - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Result", frame)
        cv2.imshow("Mask", mask)

    except Exception as e:
        print(f"Connection Error: {e}")
        continue


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.destroyAllWindows()
