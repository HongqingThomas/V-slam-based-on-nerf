import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

file_name = "save_dirct/000000_10.png"
image = cv.imread(file_name, cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR)
print("image:", image)
print(np.min(image), np.max(image))
cv.imshow("Image",image)
cv.waitKey(0)
cv.destroyAllWindows()

# img  = mpimg.imread(file_name)
# print(img.mean())
# plt.imshow(img)
# plt.show()
