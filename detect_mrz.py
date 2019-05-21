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

	def recognize(self, image_string):
		nparr = np.fromstring(image_string, np.uint8)
		img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		image = imutils.resize(img_np, height=600)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)
		threshold = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.sqKernel)
		threshold = cv2.erode(threshold, None, iterations=4)
		p = int(image.shape[1] * 0.05)
		threshold[:, 0:p] = 0
		threshold[:, image.shape[1] - p:] = 0
		contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		for c in contours:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			crWidth = w / float(gray.shape[1])
			if ar > 6 and crWidth > 0.77:
				pX = int((x + w) * 0.02)
				pY = int((y + h) * 0.02)
				(x, y) = (x - pX, y - pY)
				(w, h) = (w + pX*2, h + pY*2)
				roi = image[y:y + h, x:x + w].copy()
				img = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
				img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				treshold = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				break
				
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
		text = pytesseract.image_to_string(treshold,lang='eng', config=r'--oem 0 --psm 11 ')
		return text

	def __call__(self, image_path):
		text = self.recognize(image_path)
		return text

	@staticmethod
	def apply(image_path):
		if getattr(MRZRecognizer, '__instance__', None) is None:
			MRZRecognizer.__instance__ = MRZRecognizer()
		return MRZRecognizer.__instance__(image_path)
