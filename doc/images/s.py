import cv2
import numpy as np

b = cv2.imread('clustered.png')
b = cv2.resize(b, (1000, 750), interpolation=cv2.INTER_CUBIC)

a = cv2.imread('clustering.jpg')

cv2.imwrite('hola.png', np.hstack((a,b)))