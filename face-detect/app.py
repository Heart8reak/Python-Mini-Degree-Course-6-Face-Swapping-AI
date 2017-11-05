import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('obama.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 8)
#print len(faces)
if len(faces) != 2:
	sys.exit('Please input an image with EXACTLY 2 faces!')

x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

face1 = cv2.resize(face1, (w2, h2))
face2 = cv2.resize(face2, (w1, h1))

# face 1
img[y1:y1+h1, x1:x1+w1] = face2

# face 2 
img[y2:y2+h2, x2:x2+w2] = face1

cv2.imshow('swap', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


