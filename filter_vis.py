import cv2
import numpy as np
color_image = cv2.imread('/home/jazz-lab/Documents/nice-slam/Datasets/Highbayparking2/color/306.jpg')
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
depth_image = cv2.imread('/home/jazz-lab/Documents/nice-slam/Datasets/Highbayparking2/depth/306.png',cv2.IMREAD_UNCHANGED)
color_image = color_image / 255.
depth_image = depth_image.astype(np.float32) / 1000.

img = color_image
img_cpu = img
img_cpu = np.float32(img_cpu)
gray_img = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2GRAY)  #1

gray_blurred_img = cv2.GaussianBlur(gray_img,ksize=(5,5),sigmaX=0)  #2

grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)

# # abs_grad_x = cv2.convertScaleAbs(grad_x)
# # abs_grad_y = cv2.convertScaleAbs(grad_y)
# grad = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, .5, 0)  #4

grad_color = cv2.addWeighted(grad_x, .5, grad_y, .5, 0)  #4
# print("depthgrad", grad)
# print("max", np.amax(grad))
# cv2.imshow('gradient',abs(grad))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = depth_image
img_cpu = img
img_cpu = np.float32(img_cpu)
# gray_img = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2GRAY)  #1

gray_blurred_img = cv2.GaussianBlur(img_cpu,ksize=(5,5),sigmaX=0)  #2
# grad_depth = cv2.Laplacian(gray_blurred_img, cv2.CV_64F)

grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)

# # abs_grad_x = cv2.convertScaleAbs(grad_x)
# # abs_grad_y = cv2.convertScaleAbs(grad_y)
# grad = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, .5, 0)  #4

grad_depth = cv2.addWeighted(grad_x, .5, grad_y, .5, 0)  #4

grad = np.abs(np.multiply(grad_color, grad_depth))
grad_log = np.log(grad)
# print(grad.shape)
# np.set_printoptions(threshold=np.inf)
# # print("depthgrad", grad)
# print("max", np.sort(grad, axis = None,kind = 'quicksort')[::-1][50000])
show_grad = (grad/np.amax(grad)) * 255.0
show_log_grad = (grad_log/np.amax(grad_log)) * 255.0
show_color = color_image
print(np.amax(grad_log))
cv2.imshow('gradient0',show_grad)
cv2.imshow('gradient1',show_log_grad)
cv2.imshow('gradient2',show_color)
cv2.imshow('depth',(grad_depth))
cv2.imshow('color',(grad_color))
cv2.waitKey(0)
cv2.destroyAllWindows()