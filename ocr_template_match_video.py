# USAGE
# python ocr_template_match.py --picamera -1 --reference ocr_a_reference.png

# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
from imutils.video import VideoStream
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import datetime
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-r", "--reference", type=str, default='ocr_a_reference.png',
	help="path to reference OCR-A image")
args = vars(ap.parse_args())

# define a dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# cv2.THRESH_BINARY（黑白二值）; cv2.THRESH_BINARY_INV（黑白二值反转）
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
'''
cv2.RETR_EXTERNAL表示只检测外轮廓
cv2.RETR_LIST检测的轮廓不建立等级关系
cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
cv2.RETR_TREE建立一个等级树结构的轮廓。

cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
'''
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi

######################################################################

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)


# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	# load the input image, resize it, and convert it to grayscale

	# draw the timestamp on the frame
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.35, (0, 0, 255), 1)

	# show the frame
	cv2.imshow("Frame", frame)

	image = frame.copy()
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	orig = imutils.resize(orig, height=500)
	gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	screenCnt = []

	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.05 * peri, True)
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)
			keepDims = w > 200 and h > 100
			if keepDims:
				screenCnt = approx
				break

	# ("STEP 2: Find contours of paper")
	if len(screenCnt):
		cv2.drawContours(orig, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Camera", orig)
	if  len(screenCnt):
		warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

		image = imutils.resize(warped, width=300)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		'''
		cv2.MORPH_OPEN	开运算(open) ,先腐蚀后膨胀的过程。开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的 同时并不明显改变其面积。
		cv2.MORPH_CLOSE	闭运算(close)，先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。
		cv2.MORPH_GRADIENT	形态学梯度(morph-grad)，可以突出团块(blob)的边缘，保留物体的边缘轮廓。
		cv2.MORPH_TOPHAT	顶帽(top-hat)，将突出比原轮廓亮的部分。
		cv2.MORPH_BLACKHAT	黑帽(black-hat)，将突出比原轮廓暗的部分。
		'''

		# apply a tophat (whitehat) morphological operator to find light
		# regions against a dark background (i.e., the credit card numbers)
		tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

		# compute the Scharr gradient of the tophat image, then scale
		# the rest back into the range [0, 255]
		gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
			ksize=-1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
		gradX = gradX.astype("uint8")

		# apply a closing operation using the rectangular kernel to help
		# cloes gaps in between credit card number digits, then apply
		# Otsu's thresholding method to binarize the image
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

		thresh = cv2.threshold(gradX, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# apply a second closing operation to the binary image, again
		# to help close gaps between credit card number regions
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

		# find contours in the thresholded image, then initialize the
		# list of digit locations
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		locs = []

		# loop over the contours
		for (i, c) in enumerate(cnts):
			# compute the bounding box of the contour, then use the
			# bounding box coordinates to derive the aspect ratio
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)

			# since credit cards used a fixed size fonts with 4 groups
			# of 4 digits, we can prune potential contours based on the
			# aspect ratio
			if ar > 2.5 and ar < 4.0:
				# contours can further be pruned on minimum/maximum width
				# and height
				if (w > 40 and w < 55) and (h > 10 and h < 20):
					# append the bounding box region of the digits group
					# to our locations list
					locs.append((x, y, w, h))

		# sort the digit locations from left-to-right, then initialize the
		# list of classified digits
		locs = sorted(locs, key=lambda x:x[0])
		output = []

		# loop over the 4 groupings of 4 digits
		for (i, (gX, gY, gW, gH)) in enumerate(locs):
			# initialize the list of group digits
			groupOutput = []

			# extract the group ROI of 4 digits from the grayscale image,
			# then apply thresholding to segment the digits from the
			# background of the credit card
			group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
			group = cv2.threshold(group, 0, 255,
				cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			# detect the contours of each individual digit in the group,
			# then sort the digit contours from left to right
			digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			digitCnts = imutils.grab_contours(digitCnts)
			digitCnts = contours.sort_contours(digitCnts,
				method="left-to-right")[0]

			# loop over the digit contours
			for c in digitCnts:
				# compute the bounding box of the individual digit, extract
				# the digit, and resize it to have the same fixed size as
				# the reference OCR-A images
				(x, y, w, h) = cv2.boundingRect(c)
				roi = group[y:y + h, x:x + w]
				roi = cv2.resize(roi, (57, 88))

				# initialize a list of template matching scores
				scores = []

				# loop over the reference digit name and digit ROI
				for (digit, digitROI) in digits.items():
					# apply correlation-based template matching, take the
					# score, and update the scores list
					result = cv2.matchTemplate(roi, digitROI,
						cv2.TM_CCOEFF)
					(_, score, _, _) = cv2.minMaxLoc(result)
					scores.append(score)

				# the classification for the digit ROI will be the reference
				# digit name with the *largest* template matching score
				groupOutput.append(str(np.argmax(scores)))

			# draw the digit classifications around the group
			cv2.rectangle(image, (gX - 5, gY - 5),
				(gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
			cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

			# update the output digits list
			output.extend(groupOutput)

		# # display the output credit card information to the screen
		# print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
		# print("Credit Card #: {}".format("".join(output)))
		cv2.imshow("Card", image)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()