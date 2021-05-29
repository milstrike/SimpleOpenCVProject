import cv2 as cv
import imutils as im

faces = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes = cv.CascadeClassifier('haarcascade_eye.xml')

imgsrc = cv.imread('foto_deteksi.jpg')
imgres = im.resize(imgsrc, width=500)
imggrey = cv.cvtColor(imgres, cv.COLOR_BGR2GRAY)

faceDetection = faces.detectMultiScale(imggrey, 1.3, 1)

totalFaces = 0
for(x, y, w, h) in faceDetection:
    totalFaces = totalFaces + 1
    cv.rectangle(imgres, (x,y), (x+w, y+h), (255,0,0), 2)

print(totalFaces)

cv.imshow('HASIL AKHIR', imgres)
cv.waitKey(0)
cv.destroyAllWindows()