import cv2
import numpy as np
import os
import sys

np.set_printoptions(threshold=sys.maxsize)


def countImagesVertical(path: str, h1: int, w1: int, h2: int, w2: int):
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 190, 255, cv2.THRESH_BINARY)
    h = int((h1 + h2) / 2 + 1)
    sumArr = 0
    for i in range(w1, w2):
        sumArr += thresh[h][i]
    if int(sumArr / (w2 - w1)) < (0.1 * 255):
        return np.array([[h - 8, w2], [h + 8, w2]])
    else:
        return np.array([])


def countImagesHorizontal(path: str, h1: int, w1: int, h2: int, w2: int):
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 190, 255, cv2.THRESH_BINARY)
    h = 0
    w = int((w1 + w2) / 2 + 1)
    sumArr = 0
    if len(countImagesVertical(path, h1, w1, h2, w2)) > 0:
        h = int((h1 + h2) / 2 + 1)
    else:
        h = h1
    for i in range(h, h2):
        sumArr += thresh[i][w]

    ave = sumArr / (h2 - h)
    if 0 <= ave <= 5:
        return np.array([[h2, w - 8], [h2, w + 8]])
    else:
        return np.array([])


def findMinGlobal(path: str):
    # image = cv2.imread(path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    # for i in range(9, 1000):
    #     for j in range(100, 1179):
    #         if thresh[j][i] == 255:
    #             return j, i
    return 209, 9


def findMaxGlobal(path: str):
    # image = cv2.imread(path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    # for i in range(999, -1, -1):
    #     for j in range(1178, -1, -1):
    #         if thresh[j][i] == 255:
    #             return j, i
    return 1169, 990


def remove_pixel(input_image, size):
    # Convert RGB to grayscale:
    grayscaleImage = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # Convert the BGR image to HSV:
    hsvImage = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    # Create the HSV range for the red ink:
    # 'red1': [[180, 255, 255], [159, 50, 70]]
    # 'red2': [[9, 255, 255], [0, 50, 70]]
    lower_red1 = np.array([0, 50, 70])
    upper_red1 = np.array([9, 255, 255])

    lower_red2 = np.array([159, 50, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_black1 = np.array([0, 0, 0])
    upper_black1 = np.array([180, 255, 30])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsvImage, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsvImage, lower_red2, upper_red2)

    # Threshold the HSV image to get only black colors
    maskBlack = cv2.inRange(hsvImage, lower_black1, upper_black1)
    for i in range(size):
        for j in range(size):
            if i < 30 and 100 < j < 220:
                maskBlack[i][j] = 0
            else:
                if 30 <= i < (size - 20):
                    maskBlack[i][j] = 0
    mask = cv2.bitwise_or(mask1, mask2)
    maskFinal = cv2.bitwise_or(mask, maskBlack)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # # Perform closing:
    bluepenMask = cv2.morphologyEx(maskFinal, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)
    #
    # Add the white mask to the grayscale image:
    result = cv2.add(grayscaleImage, bluepenMask)
    return result


def showImage(path: str):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 190, 255, cv2.THRESH_BINARY)
    imS = cv2.resize(thresh, (1000, 1179))
    cv2.imshow("output", imS)
    cv2.waitKey()
    cv2.destroyWindow()


def cropImage(path: str, imagesMaxMin: object, imagesVertical: object, imagesHorizontal: object, size: int):
    image = cv2.imread(path)
    if len(imagesVertical) == 0:
        imgCrop1 = image[imagesMaxMin[0][0]:imagesHorizontal[0][0], imagesMaxMin[0][1]:imagesHorizontal[0][1]]
        imS1 = cv2.resize(imgCrop1, (size, size))
        cv2.imwrite(f'ant_v1/ant_{path[4::]}', remove_pixel(imS1, size))

        imgCrop2 = image[imagesMaxMin[0][0]:imagesHorizontal[0][0], imagesHorizontal[1][1]:imagesMaxMin[1][1]]
        imS2 = cv2.resize(imgCrop2, (size, size))
        cv2.imwrite(f'pos_v1/pos_{path[4::]}', remove_pixel(imS2, size))
    else:
        if len(imagesHorizontal) == 0:
            imgCrop1 = image[imagesMaxMin[0][0]:imagesVertical[0][0], imagesMaxMin[0][1] + 160:imagesMaxMin[1][1] - 170]
            imS1 = cv2.resize(imgCrop1, (size, size))
            cv2.imwrite(f'ant_v1/ant_{path[4::]}', remove_pixel(imS1, size))

            imgCrop2 = image[imagesVertical[1][0]:imagesMaxMin[1][0], imagesMaxMin[0][1] + 160:imagesMaxMin[1][1] - 170]
            imS2 = cv2.resize(imgCrop2, (size, size))
            cv2.imwrite(f'pos_v1/pos_{path[4::]}', remove_pixel(imS2, size))
        else:
            imgCrop1 = image[imagesMaxMin[0][0]:imagesVertical[0][0], imagesMaxMin[0][1] + 160:imagesMaxMin[1][1] - 170]
            imS1 = cv2.resize(imgCrop1, (size, size))
            cv2.imwrite(f'ant_v1/H&N ant_{path[4::]}', remove_pixel(imS1, size))

            imgCrop2 = image[imagesVertical[1][0]:imagesMaxMin[1][0],
                       imagesMaxMin[0][1] + 100:imagesHorizontal[0][1] - 105]
            imS2 = cv2.resize(imgCrop2, (size, size))
            cv2.imwrite(f'ant_v1/ant_{path[4::]}', remove_pixel(imS2, size))

            imgCrop3 = image[imagesVertical[1][0]:imagesMaxMin[1][0],
                       imagesHorizontal[1][1] + 100:imagesMaxMin[1][1] - 105]
            imS3 = cv2.resize(imgCrop3, (size, size))
            cv2.imwrite(f'pos_v1/pos_{path[4::]}', remove_pixel(imS3, size))


def runCropAuto(path: str):
    print(path[4::])
    h1, w1 = findMinGlobal(path)
    h2, w2 = findMaxGlobal(path)
    cropImage(path, np.array([[h1, w1], [h2, w2]]), countImagesVertical(path, h1, w1, h2, w2),
              countImagesHorizontal(path, h1, w1, h2, w2), 320)


if __name__ == '__main__':
    ids = sorted(os.listdir('jpg'))
    for i in ids:
        runCropAuto(f"jpg/{i}")
