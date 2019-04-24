from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import pytesseract

class MRZRecognizer():
	def __init__(self):
		self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
		self.sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
		
	def image_smoothening(self,img):
		img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret1, th1 = cv2.threshold(img_grey, 180, 255, cv2.THRESH_BINARY)
		ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		blur = cv2.GaussianBlur(th2, (1, 1), 0)
		ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return th3

	def recognize(self, image_string):
		nparr = np.fromstring(image_string, np.uint8)
		img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.
		image = imutils.resize(img_np, height=600)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)
		cv2.imshow("blackhat", blackhat)
		gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, self.rectKernel)
		thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.sqKernel)
		thresh = cv2.erode(thresh, None, iterations=4)
		p = int(image.shape[1] * 0.05)
		thresh[:, 0:p] = 0
		thresh[:, image.shape[1] - p:] = 0
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			crWidth = w / float(gray.shape[1])
			if ar > 5 and crWidth > 0.75:
				pX = int((x + w) * 0.03)
				pY = int((y + h) * 0.03)
				(x, y) = (x - pX, y - pY)
				(w, h) = (w + (pX * 2), h + (pY * 2))
				roi = image[y:y + h, x:x + w].copy()
				img = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
				img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				filtered = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
				kernel = np.ones((1, 1), np.uint8)
				opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
				closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
				img = self.image_smoothening(img)
				or_image = cv2.bitwise_or(img, filtered)
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				break
				
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
		text = pytesseract.image_to_string(img_grey,lang='eng', config=r'--oem 0 --psm 11 ')
		return text

	def __call__(self, image_path):
		text = self.recognize(image_path)
		return text

	@staticmethod
	def apply(image_path):
		if getattr(MRZRecognizer, '__instance__', None) is None:
			MRZRecognizer.__instance__ = MRZRecognizer()
		return MRZRecognizer.__instance__(image_path)
