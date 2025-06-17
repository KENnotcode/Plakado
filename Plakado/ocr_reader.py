import easyocr
import cv2

class OCRReader:
    def __init__(self, lang=["en"], gpu=False):
        self.reader = easyocr.Reader(lang, gpu=gpu)

    def read(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        return results[0][-2] if results else ""
