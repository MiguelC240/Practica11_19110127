import cv2
 

 # Leer en dos imágenes como imágenes en escala de grises
img1 = cv2.imread("Crash.jpeg")
img2 = cv2.imread("Crash_rec.jpeg")
 
 # Crear detector de características ORB y descriptor
orb = cv2.ORB_create()
 # Detecta características y descriptores en dos imágenes
keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
keypoint2, descriptor2 = orb.detectAndCompute(img2, None)

 # Consigue un objeto violento
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 # Use el comparador para hacer coincidir la similitud de los dos descriptores
maches = bf.match(descriptor1, descriptor2)
 # Ordenar por similitud
maches = sorted(maches, key=lambda x: x.distance)
 # Empates
img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, maches[: 30], img2, flags=2)
 
cv2.imshow("Imagenes juntas", img3)
cv2.waitKey()


########################### OBJETIVO 2 #############################
cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if ret == False: break
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
