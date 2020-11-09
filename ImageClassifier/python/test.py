import numpy as np
import cv2
import RawNN2

classifier = RawNN2.Classifier()

img = cv2.imread('6779.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (180,180))

score = classifier.Run(img)
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)