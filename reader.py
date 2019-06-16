import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import re
import os

import DetectChars
import PossiblePlate



class reader(object):
	"""docstring for ClassName"""
	def __init__(self, img):
		self.img = img


	def preprocess_txt(self, text):

		regex = re.compile('[^a-z0-9A-Z ]')

		text = regex.sub('', text)
		i = 0
		while i < len(text) and not text[i].isdigit():
			i += 1

		text = text[i:]

		return text

	def similar_digit(self, a):
		if a.isdigit():
			return a
		if a == 'O':
			return '0'
		if a == 'I':
			return '1'
		if a == 'E':
			return '3'
		if a == 'H' or a == 'K' or a == 'L':
			return '4'
		if a == 'S':
			return '5'
		if a == 'G':
			return '6'
		if a == 'T':
			return '7'
		if a == 'B':
			return '8'
		if a == 'Q':
			return '9'

		return '2'

	def similar_char(self, a):
		if a == '0':
			return 'O'
		if a == '1':
			return 'I'
		if a == '2':
			return 'Z'
		if a == '3':
			return 'E'
		if a == '4':
			return 'H'
		if a == '5':
			return 'S'
		if a == '6':
			return 'G'
		if a == '7':
			return 'T'
		if a == '8':
			return 'B'
		if a == '9':
			return 'G'

		return a

	def format_plate(self, s):
		plate = list(s)

		if len(plate) > 2:
			if not plate[0].isdigit():
				plate[0] = self.similar_digit(plate[0])
			if not plate[1].isdigit():
				plate[1] = self.similar_digit(plate[1])
			if not plate[2].isalpha():
				plate[2] = self.similar_char(plate[2])

			i = 3

			while i < len(s) and not plate[i].isdigit():
				i += 1

			while i < len(s):
				if not plate[i].isdigit():
					plate[i] = self.similar_digit(plate[i])
				i += 1


		i = 0
		while i < len(plate):
			if plate[i] == 'X':
				plate[i] = 'K'
			elif plate[i] == 'Q':
				plate[i] = 'O'
			elif plate[i] == 'W':
				plate[i] = 'Y'
			i += 1


		a = ''.join(plate)

		return a

	def ocr(self):

		gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		otsu = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

		text_gry = pytesseract.image_to_string(gray)
		text_otsu = pytesseract.image_to_string(otsu)

		text_gry = self.preprocess_txt(text_gry)
		text_otsu = self.preprocess_txt(text_otsu)

		if len(text_gry) > len(text_otsu):
			return(text_gry)

		return(text_otsu)

	def read_deep(self):

		plate = PossiblePlate.PossiblePlate(self.img)

		listOfPossiblePlates = DetectChars.detectCharsInPlates([plate])

		if len(listOfPossiblePlates) == 0:                          # if no plates were found
		    # print("\nno license plates were detected\n")  # inform user no plates were found
		    return ''
		else:                                                       # else
		            # if we get in here list of possible plates has at leat one plate

		            # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
		    listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

		            # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
		    licPlate = listOfPossiblePlates[0]

		    plate = licPlate.strChars

		return self.format_plate(self.preprocess_txt(plate))

	def read(self):

		txt1 = self.ocr()
		txt2 = self.read_deep()


		if len(txt1) > len(txt2):
			return txt1

		return txt2