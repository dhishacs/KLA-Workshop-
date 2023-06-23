import json
import sys
import cv2
import numpy as np

with open('Level_1_Input_Data\input.json', "r") as file:
    l1 = json.load(file)

#print("Type of JSON: ", type(l1)) - type is dictionary

'''for i in l1:
    print(i, l1[i])
 prints the content of json file as key and value'''

 #using openCV to read the image

img1 = cv2.imread("Level_1_Input_Data\wafer_image_1.png", cv2.IMREAD_ANYCOLOR)
img2 = cv2.imread("Level_1_Input_Data\wafer_image_2.png", cv2.IMREAD_ANYCOLOR)
img3 = cv2.imread("Level_1_Input_Data\wafer_image_3.png", cv2.IMREAD_ANYCOLOR)
img4 = cv2.imread("Level_1_Input_Data\wafer_image_4.png", cv2.IMREAD_ANYCOLOR)
img5 = cv2.imread("Level_1_Input_Data\wafer_image_5.png", cv2.IMREAD_ANYCOLOR)


'''while True:
    cv2.imshow("Chip 1", img1)
    cv2.waitKey(0)
    sys.exit()

cv2.destroyAllWindows()
WORKING - DISPLAYS IMAGE
-----
diff = cv2.subtract(img1, img2)
b, g, r = cv2.split(diff)
if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("Colours are equal")
else:
    print("Colours of the images are different")
    cv2.imshow("Difference",diff)
    cv2.waitKey(0)
 - NOT THE CORRECT ONE - WORKING ON ANOTHER ONE
-----
def mse(img1, img2):
    err = np.sum(img1.astype("float") - img2.astype("float"))**2
    err /= float(img1.shape[0] * img1.shape[1])
    return err

print(mse(img1, img2))
- IDENTIFYING THE MSE - MEAN SQ ERROR BTW 2 IMAGES,
BUT IS IT OF ANY USE?? 
'''

def img_diff(img1, img2):
    if(img1.shape == img2.shape):
        #computing the abs difference
        diff = cv2.absdiff(img1, img2)

        #splitting the diff img into different colour channels
        b,g,r = cv2.splt(diff)

        #combining the colour channels into a single image
        zeros = np.zeros(diff.shape[:2], dtype=np.uint8)
        diff = cv2.merge((b,zeros,zeros))
        return diff

    else:
        raise ValueError("Image must of the same size")

def circle_diff(diff, img):
    ret, thresh = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY)
    #contours - are the line joining all the points in the 
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


