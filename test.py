import cv2
import numpy as np

img = cv2.imread('./darkInCars.jpeg')
cv2.imwrite('./test.jpg', img)