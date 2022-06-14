import numpy as np
import cv2
import time
from deslant_img import deslant_img
from scipy.ndimage import interpolation as inter


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

# slanting and skewness removal for one word
# deslanting and deskew for  one word


if __name__ == "__main__":
    start_time = time.time()
    img = cv2.imread('0_2.jpg')
    angle, img = correct_skew(img)
    print('Skew angle: ', angle)
    cv2.imwrite("./rotated.png", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # rotation of the image based on skew angle
    img = img.astype(np.uint8)
    res = deslant_img(img)
    cv2.imwrite("./corrected.png", res.img)
    #img = rotate(img, skew_angle_hough_transform(res.img), cval=1)

    print("--- %s seconds ---" % (time.time() - start_time))

    # deslanting
