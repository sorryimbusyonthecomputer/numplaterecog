import cv2
import numpy as np
import easyocr
import time
import imutils
import threading

reader = easyocr.Reader(['en'])
words = []
word_count = {}
confident_words = []

def word_average():
	try:
		return sum(word_count.values()) / len(word_count)
	except ZeroDivisionError:
		print('Not enough values to calculate an average')


def word_cutoff():
	cutoff = word_average()
	for key, value in word_count.items():
		if value >= cutoff:
			print(f'{key}: {value}')


def word_counter(word):
	
	if word in word_count:
		word_count[word] += 1
	else:
		word_count[word] = 1	

def contour_and_mask(edge_img, gray_img, frame):
	keypoints = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(keypoints)
   
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
	location = None
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 10, True)
		if len(approx) == 4:
			location = approx
			break

	if location is None:
		print("No valid contour found")
		return None
	
 
	mask = np.zeros(gray_img.shape, np.uint8)
	cv2.drawContours(mask, [location], 0, 255, -1)
	plate_region = cv2.bitwise_and(frame, frame, mask=mask)
	return plate_region

def process_frame(frame, frame_number):
	grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	bfilter = cv2.bilateralFilter(grayscale_frame, 11, 17, 17)
	v = np.median(bfilter)
	lower = int(max(0, 0.7 * v))
	upper = int(max(255, 1.3 * v))
	edged = cv2.Canny(bfilter, lower, upper)


	try:	
		results = reader.readtext(contour_and_mask(edged.copy(), grayscale_frame, frame))
		for bbox, text, confidence in results:
			print(f'Detected text: "{text}" with confidence: {confidence:.2f}')
			if confidence > 0.22:
				word_counter(text)
				if confidence > 90:
					confident_words.append(text)
	except Exception as e:
		pass	

	print(f'Finished processing frame: {frame_number}')


def capture():
	
	capture = cv2.VideoCapture(0)
	if not capture.isOpened():
		print('Cannot open webcam')
		exit()
	
	frame_number = 0

	try:
		while True:
			ret, frame = capture.read()
			if not ret:
				print('Failed to grab frame')
				break
			
			frame_number += 1
			
			if frame_number % 10 == 0:
				threading.Thread(target=process_frame, args=(frame, frame_number)).start()

			cv2.imshow('WebcamFeed', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			if cv2.waitKey(1) & 0xff == ord('l'):
				print(word_count)

			time.sleep(0.03)
	
	except KeyboardInterrupt:
		print('Exited by user')

	except Exception as e:
		print(f'Exception: {e}')

	finally:
		capture.release()
		cv2.destroyAllWindows()
		print('Webcam Closed')
		word_cutoff()
		print(confident_words)
capture()


