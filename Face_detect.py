import cv2 as cv
from skinDetector import SkinDetector
import numpy as np
import matplotlib.pyplot as plt

f = open('Contact.txt', 'r')
print(f)
results = []
y = []
x = []
k = 0
mass_i = []

for line in f:
    k += 1
    line = line.split(',')
    mass_i.append(k)
    results.append([k, int(line[1])])


for i in results:
    x.append(i[0])
    y.append(i[1])
f.close()

def face_detection(image):

    #Исходное изображение
    #cv.imshow('img', image)

    #Определение лица с помощью каскада Хаара
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=9)
    #print(f'Number of faces found = {len(faces_rect)}')

    #Выделение лица
    for (x,y,w,h) in faces_rect:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cropped = image[y:y + h, x:x + w]
        if cv.waitKey(20) and 0xFF == ord('d'):
            break
    return cropped

    #Полное изображение с выделенным лицом
    #cv.imshow('Detected face', image)

    cv.waitKey(1)


VPG_mass = []

def face_detection_video(video):
    while True:
        isTrue, frame = video.read()

        #Вычисление VPG сигнала
        mass = face_detection(frame)
        R = 0
        G = 0
        B = 0
        column_sums = np.sum(mass, axis=0)
        for i, column_sum in enumerate(column_sums):
            B += column_sum[0]
            G += column_sum[1]
            R += column_sum[2]
        VPG = 1000*G/(R+B)
        VPG_mass.append(VPG)


        detector = SkinDetector(face_detection(frame))
        detector.find_skin()
        detector.show_all_images()
        if cv.waitKey(20) and 0xFF == ord('d'):
            break
    video.release()
    cv.destroyAllWindows()

video = cv.VideoCapture('Gorelova_2.avi')
image = cv.imread('Test.jpg')

face_detection_video(video)

FPG_plot = plt.plot(x, y, c="g")
VPG_plot = plt.plot(mass_i, VPG_mass, c="r")
plt.xlabel('ms')
plt.ylabel('VPG')
plt.show()




