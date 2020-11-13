import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    src = cv2.imread('res/lakers25601440.jpeg')
    paddle = cv2.imread('output/paddlelite_bgr_from_nv21_neon.jpg')
    our = cv2.imread('output/our_bgr_from_nv21_neon.jpg')

    src = src.reshape((1, -1))
    paddle = paddle.reshape((1, -1))
    our = our.reshape((1, -1))

    print(cosine_similarity(src, paddle))
    # [[0.99951482]]
    print(cosine_similarity(src, our))
    # [[0.99235254]]

