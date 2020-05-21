import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
import math

dsiac_image = "/home/franciscoAML/Documents/DSIAC/Five_Classes/DSIAC_3_0/Version7/Single_Channel/yolov3/Francisco/images/cegr01923_0001_51.png"

def channel_test():
    orig = cv2.imread(dsiac_image)
    #fig = plt.figure()
    #plt.imshow(orig)
    #plt.title("Original Image")
    #plt.savefig("/home/franciscoAML/Documents/DSIAC/Five_Classes/DSIAC_3_0/Version7/Single_Channel/yolov3/Francisco/Image_augmentations/Original_image")


    # Read Image
    img = cv2.imread(dsiac_image)[:,:,0]                                                                                    # Convert Image into Single Channel by decomposition
    #fig1 = plt.figure()
    #plt.imshow(img)
    #plt.title("Manual Channel Decomposition")
    #plt.savefig("/home/franciscoAML/Documents/DSIAC/Five_Classes/DSIAC_3_0/Version7/Single_Channel/yolov3/Francisco/Image_augmentations/Manual_Channel_Decomposition")

    img2 = cv2.imread(dsiac_image,0)                                                                                        # Convert Image into Single Channel by cv2
    #fig2 = plt.figure()
    #plt.imshow(img2)
    #plt.title("OpenCV Automatic Decomposition")
    #plt.savefig("/home/franciscoAML/Documents/DSIAC/Five_Classes/DSIAC_3_0/Version7/Single_Channel/yolov3/Francisco/Image_augmentations/CV_Automatic_Decomposition")
    return orig, img, img2

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def augment_hsv(img, hgain=0.2, sgain=0.678, vgain=0.5):
    orig_img = np.copy(img)
    #img = cv2.merge((img,img,img)) # Combine Image into a Three Channel Image
    x = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
    np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    hue_image = img
    b,g,r = cv2.split(img)
    print("B vs G:", mean_squared_error(b,g))
    print("B vs R:", mean_squared_error(b,r))
    print("G vs R:", mean_squared_error(g,r))
    print("Three Channel Image MSE:", mean_squared_error(orig_img[:,:,0], hue_image[:,:,0]))
    img = b
    return img

def left_right_flip(img):
    lr_flip = True
    img = np.fliplr(img)
    return img
def up_down_flip(img):
    img = np.flipud(img)
    return img
def random_affine(img, degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates


    return img

def main():
    original_image, channel_img, cv2_img = channel_test()
    print(original_image.shape)
    # Test Letterbox
    img, ratio, pad = letterbox(original_image, 640, auto=True, scaleup=True)
    orig_img = img
    img = augment_hsv(img)
    #img = left_right_flip(img)
    #img = up_down_flip(img)
    #print(img.shape, ratio)
    #img = random_affine(img)

    # Calculate MSE
    #mse = mean_squared_error(orig_img, img)
    #print("MSE:", mse)
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(orig_img)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.title("Augmented Image")
    plt.show()
    #plt.savefig("/home/franciscoAML/Documents/DSIAC/Five_Classes/DSIAC_3_0/Version7/Single_Channel/yolov3/Francisco/Image_augmentations/Random_Affine")


main()