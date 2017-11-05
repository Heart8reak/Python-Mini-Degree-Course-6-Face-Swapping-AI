import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('face.jpg')
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for face in faces:
    x, y, w, h = face
    cv2.rectangle(img, (x,y), (x+w, y+h), ( 0,255,0), 2)

    color_face = img[y:y+h, x:x+w]
    gray_face = gray[y:y+h, x:x+w]

eyes = eye_cascade.detectMultiScale(gray_face, 1.075, 3)
for eye in eyes:
    x, y, w, h = eye
    cv2.rectangle(color_face, (x, y), (x+w, y+h), (0,0,255), 1)


cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
