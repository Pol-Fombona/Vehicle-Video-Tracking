# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS

from tracker import Tracker

import numpy as np
import argparse
import imutils
import time
import cv2


"""  GLOBAL VARIABLES """
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIDTH = 480
HISTORY = 500
THS = 300
SHADOWS = True
# Preprocessing
KERNEL_SIZE = 7
ERODE_KERNEL = 3
MIN_AREA = 5000
# Colors
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Kernel for morphological op.
erode_kernel = np.ones((ERODE_KERNEL, ERODE_KERNEL), np.uint8)
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
# Background object subtractor
backgroundobject = cv2.createBackgroundSubtractorMOG2(
    history=HISTORY, varThreshold=THS, detectShadows=SHADOWS)
# Tracker object
tracker = Tracker(maxDissapeared=20)


def detection(frame):
    # Detects objects that are moving and draw their contour
    fgmask = backgroundobject.apply(frame)

    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    fgmask = cv2.erode(fgmask, erode_kernel, iterations=2)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()
    centroids = list()

    # loop over each contour found in the frame.
    for cnt in contours:
        # We need to be sure about the area of the contours i.e. it should be higher than 400 to reduce the noise.
        if cv2.contourArea(cnt) >= MIN_AREA:
            # Accessing the x, y and height, width of the cars
            x, y, width, height = cv2.boundingRect(cnt)
            # Here we will be drawing the bounding box on the cars
            cv2.rectangle(frameCopy, (x, y),
                          (x + width, y + height), RED, 2)
            # Find middle point for every bbox
            middle_point = (x + width//2, y + height//2)
            
            # Save the centroid found in the frame
            centroids.append(middle_point)

            # Then with the help of putText method we will write the 'Car detected' on every car with a bounding box
            cv2.putText(frameCopy, 'Car Detected', (x, y-10),
                        FONT, 0.3, GREEN, 1, cv2.LINE_AA)


    objects = tracker.update(centroids = centroids)
    # Draw the IDs tracked
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frameCopy, text, (int(centroid[0]), int(centroid[1] - 10)),
            FONT, 2, RED, 2, cv2.LINE_AA)
        cv2.circle(frameCopy, (int(centroid[0]), int(centroid[1])), radius=5, color=RED, thickness=-1)

    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)

    stacked = np.hstack((foregroundPart, frameCopy))
    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars',
               cv2.resize(stacked, None, fx=0.65, fy=0.65))

    cv2.waitKey(0)


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video file")
    args = vars(ap.parse_args())

    # start the file video stream thread and allow the buffer to start to fill
    fvs = FileVideoStream(args["video"]).start()
    time.sleep(1.0)

    # start the FPS timer
    fps = FPS().start()

    # loop over frames from the video file stream
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = fvs.read()
        #frame = imutils.resize(frame, width=WIDTH)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = np.dstack([frame, frame, frame])

        detection(frame)

    # Out of the loop, clean space
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()
