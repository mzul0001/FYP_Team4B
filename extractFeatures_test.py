import unittest
import cv2
from imageProcessing import enhanceImage
from extractFeatures import extractFeatures


class TestExtractFeatures(unittest.TestCase):
    def test_extractFeatures(self):
        img = cv2.imread('lena512color.tiff')
        enhanced_img = enhanceImage(img)
        features = extractFeatures(enhanced_img)
        self.assertIsNotNone(features)


if __name__ == '__main__':
    unittest.main()