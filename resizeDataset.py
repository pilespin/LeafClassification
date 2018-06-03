
import numpy as np
import cv2


def resize(img):

	height, width = img.shape[:2]
	width_final = 100
	# height_final = 100

	ratio_width = width_final / width

	print (width)
	print (height)

	new = cv2.resize(img, None, fx=ratio_width, fy=ratio_width)
	background = np.zeros((100,100,4),np.uint8)

	# Merge

	cv2.imwrite('newimg.jpg', new)
	cv2.imwrite('newimg2.jpg', background)

	return new

img = cv2.imread("datasets/images/2.jpg")
if (img is None):
	print("Image not read")


resize(img)


# path_new_dataset = 'new/'
# path_dataset = 'datasets/images/'

# if not os.path.exists(path_new_dataset):
# 	os.mkdir(path_new_dataset)

# output = []
# for current in os.listdir(path_dataset):
# 	if current[0] != '.':
# 		img = cv2.imread(path_dataset + current)	
# 		img = cv2.resize(img, (100, 100)) 
# 		cv2.imwrite(path_new_dataset + current, img)

# cv2.imshow("Original image", img)
# cv2.imshow("new image", new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()