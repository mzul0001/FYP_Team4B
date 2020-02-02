import unittest
import cv2
from nn import loadModel
from imageProcessing import enhanceImage, processImage


class TestImageProcessing(unittest.TestCase):
    def test_enhanceImage(self):
        img = cv2.imread('lena512color.tiff')
        enhanced_img = enhanceImage(img)
        self.assertNotEqual(img.all(), enhanced_img.all())

    def test_processImage(self):
        model = loadModel('model.json', 'model.h5')
        img = cv2.imread('lena512color.tiff')
        _, labels = processImage(img, model)
        self.assertGreater(len(labels), 0)

        img = cv2.imread('group.jpg')
        _, labels = processImage(img, model)
        self.assertGreater(len(labels), 1)

        img = cv2.imread('faceTattoo.jpg')
        _, labels = processImage(img, model)
        self.assertEqual(len(labels), 0)

        img = cv2.imread('scenery.jpg')
        _, labels = processImage(img, model)
        self.assertEqual(len(labels), 0)


if __name__ == '__main__':
    unittest.main()
