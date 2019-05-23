from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
import matplotlib.pyplot as plt

class MRZRecognizer():
	def __init__(self):
		self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
		self.sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

	def rotate_image(self, mat, angle):
		height, width = mat.shape[:2] 
		image_center = (width/2, height/2) 
		rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
		abs_cos = abs(rotation_mat[0,0]) 
		abs_sin = abs(rotation_mat[0,1])
		bound_w = int(height * abs_sin + width * abs_cos)
		bound_h = int(height * abs_cos + width * abs_sin)
		rotation_mat[0, 2] += bound_w/2 - image_center[0]
		rotation_mat[1, 2] += bound_h/2 - image_center[1]
		rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
		return rotated_mat
		
	def find_mrz_in_countours(self, contours, image):
		for c in contours:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			crWidth = w / float(image.shape[1])
			if ar > 4 and crWidth > 0.77:
				pX = int((x + w) * 0.015)
				pY = int((y + h) * 0.015)
				(x, y) = (abs(x - pX), abs(y - pY))
				(w, h) = (w + pX*2, h + pY*2)
				roi = image[y:y + h, x:x + w].copy()
				img = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
				img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				treshold = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 4)
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				return treshold

	def grab_contours(self, image):
		contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		return contours

	def recognize(self, image_string):
		img_np = plt.imread(image_string)
		image = imutils.resize(img_np, height=600)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)
		threshold = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.sqKernel)
		threshold = cv2.dilate(threshold, None, iterations=12)
		threshold = cv2.erode(threshold, None, iterations=14)
		p = int(image.shape[1] * 0.05)
		threshold[:, 0:p] = 0
		threshold[:, image.shape[1] - p:] = 0
		contours = self.grab_contours(threshold)
		mrz = self.find_mrz_in_countours(contours, image)
		if mrz is None:
			rotated_threshold = self.rotate_image(threshold, 270)
			rotated_image = self.rotate_image(image, 270)
			contours = self.grab_contours(rotated_threshold)
			mrz = self.find_mrz_in_countours(contours, rotated_image)
		kernel = np.ones((2,2), np.uint8)
		mrz = cv2.morphologyEx(mrz, cv2.MORPH_CLOSE, kernel, iterations=2)
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
		text = pytesseract.image_to_string(mrz,lang='eng', config=r'--oem 0 --psm 11 ')
		return text

	def __call__(self, image_path):
		text = self.recognize(image_path)
		return text

	@staticmethod
	def apply(image_path):
		if getattr(MRZRecognizer, '__instance__', None) is None:
			MRZRecognizer.__instance__ = MRZRecognizer()
		return MRZRecognizer.__instance__(image_path)
