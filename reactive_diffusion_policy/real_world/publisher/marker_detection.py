import cv2
import numpy as np
from reactive_diffusion_policy.real_world.publisher import setting


def init(frame):
    RESCALE = setting.RESCALE
    return cv2.resize(frame, (0, 0), fx=1.0 / RESCALE, fy=1.0 / RESCALE)


def find_marker(frame):
    RESCALE = setting.RESCALE
    markerThresh = -13

    img_gaussian = np.int16(cv2.GaussianBlur(frame, (int(63 / RESCALE), int(63 / RESCALE)), 0))
    I = frame.astype(np.double) - img_gaussian.astype(np.double)

    # # this is the mask of the markers
    markerMask = ((np.max(I, 2)) < markerThresh).astype(np.uint8)

    # lessen the noise to smoothen the marker regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(15 / RESCALE), int(15 / RESCALE)))
    markerMask = cv2.morphologyEx(markerMask, cv2.MORPH_CLOSE, kernel)
    mask = markerMask * 255

    return mask


def marker_center(mask, frame):
    RESCALE = setting.RESCALE

    areaThresh1 = 50 / RESCALE ** 2
    areaThresh2 = 1920 / RESCALE ** 2
    MarkerCenter = []

    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours[0]) < 25:  # if too little markers, then give up
        print("Too less markers detected: ", len(contours))
        return MarkerCenter

    for contour in contours[0]:
        x, y, w, h = cv2.boundingRect(contour)
        AreaCount = cv2.contourArea(contour)
        # print(AreaCount)
        if AreaCount > areaThresh1 and AreaCount < areaThresh2 and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 2:
            t = cv2.moments(contour)
            # print("moments", t)
            # MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)
            mc = [t['m10'] / t['m00'], t['m01'] / t['m00']]
            # if t['mu11'] < -100: continue
            MarkerCenter.append(mc)
            # print(mc)
            # cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, ( 0, 0, 255 ), 2, 6);

    # 0:x 1:y
    return MarkerCenter


def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)


def warp_perspective(img):
    TOPLEFT = (175, 230)
    TOPRIGHT = (380, 225)
    BOTTOMLEFT = (10, 410)
    BOTTOMRIGHT = (530, 400)

    WARP_W = 215
    WARP_H = 215

    points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT])
    points2 = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])

    matrix = cv2.getPerspectiveTransform(points1, points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))

    return result


def init_HSR(img):
    DIM = (640, 480)
    img = cv2.resize(img, DIM)

    K = np.array(
        [[225.57469247811056, 0.0, 280.0069549918857], [0.0, 221.40607131318117, 294.82435570493794], [0.0, 0.0, 1.0]])
    D = np.array([[0.7302503082668154], [-0.18910060205317372], [-0.23997727800712282], [0.13938490908400802]])
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warp_perspective(undistorted_img)