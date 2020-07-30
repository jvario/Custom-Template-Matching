#   ------------------------------Computer Vision------------------------------
#                    ----------LAB2-Mainscript_Q2----------
#                              Name: Giannis
#                              Surname: Variozidis
#                              Email: cs141065@uniwa.gr
#                              ID: cs141065
#   ---------------------------------------------------------------------------

# import the necessary packages
import cv2


# --------------------------FUNCTIONS--------------------------

# function for sliding window overlaping the original image
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# ---------------------MAIN PROGRAM----------------------------
# load the images
image = cv2.imread('images to use/house.jpg')
temp = cv2.imread('images to use/housetemp.jpg')
# computing temp histogram
template_hist = cv2.calcHist(temp, [0], None, [256], [0, 256])
# define the window width and height
winW = temp.shape[1]
winH = temp.shape[0]
# input threshold of user
threshold = input('Give Threshod:\n')
# loop over the sliding window for each layer of the pyramid
for (x, y, window) in sliding_window(image, stepSize=20, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    image_hist = cv2.calcHist(window, [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(template_hist, image_hist, cv2.HISTCMP_INTERSECT)
    # clone the original image for applying boundies
    clone = image.copy()
    # bound Box
    im1 = cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # print in screen score
    cv2.putText(im1, str(hist_score), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    # threshold for finding matches
    if (hist_score > float(threshold)):
        im2 = cv2.rectangle(image, (x, y), (x + winW, y + winH), (255, 0, 0), 2)
        #mark them in original image
        cv2.putText(im2, str(hist_score), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # print new window
    cv2.imshow("Window", clone)
    cv2.waitKey(20)

# print final image with matches
cv2.imshow("Window", image)
cv2.waitKey(0)
