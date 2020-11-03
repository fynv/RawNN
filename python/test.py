import cv2
import RawNN

detector = RawNN.FaceDetector()

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector.Run(img)
res = detector.GetResults()
print(res)

