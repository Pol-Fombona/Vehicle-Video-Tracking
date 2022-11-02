# Import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS

from tracker import Tracker
from counter import Counter

import numpy as np
import argparse
import time
import cv2


"""  GLOBAL VARIABLES """
FONT = cv2.FONT_HERSHEY_SIMPLEX
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


roi_1 = [(120, 800), (240, 800), (120, 950), (240, 950)]
roi_2 = [(280, 800), (400, 800), (280, 950), (400, 950)]


# Counter object
counter = Counter(roi_1, roi_2)
# Tracker object
tracker = Tracker(maxDissapeared=20, minDist=40)


def track(centroids, frame):
    objects = tracker.update(centroids = centroids)
    # Draw the IDs tracked
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (int(centroid[0]), int(centroid[1] - 20)),
            FONT, 1, RED, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), radius=5, color=RED, thickness=-1)
        counter.update(centroid, objectID)

    (cars_in, cars_out) = counter.get_counter()
    return cars_in, cars_out


def detection(frame):
    # Detects objects that are moving and draw their contour
    fgmask = backgroundobject.apply(frame)

    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    fgmask = cv2.erode(fgmask, erode_kernel, iterations=2)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), RED, 2)
            # Find middle point for every bbox
            middle_point = (x + width//2, y + height//2)
            # Save the centroid found in the frame
            centroids.append(middle_point)
    
    # Draw ROIs
    fCopy = frameCopy.copy()
    cv2.rectangle(fCopy, roi_1[0], roi_1[3], GREEN, -1)
    cv2.rectangle(fCopy, roi_2[0], roi_2[3], BLUE, -1)
    frameCopy = cv2.addWeighted(fCopy, 0.2, frameCopy, 1 - 0.2, 0)

    return centroids, frameCopy



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
    number_frames = 0
    start_time = time.time()
    # loop over frames from the video file stream
    while fvs.more():

        frame = fvs.read()
        if frame is not None:
            number_frames += 1

            centroids, frame = detection(frame)
            cars_in, cars_out = track(centroids, frame)

            cv2.putText(frame, f"CARS IN: {cars_in}", (20, 40), FONT, 1.2, RED, 2, cv2.LINE_AA)
            cv2.putText(frame, f"CARS OUT: {cars_out}", (20, 80), FONT, 1.2, RED, 2, cv2.LINE_AA)
            cv2.imshow('Detected Cars', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f'\n[*] Key "{chr(key)}" pressed. Exiting... \n')
                exit(0)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n[*] Elapsed time: {elapsed_time:2f} \
        Frames per second: {number_frames//int(elapsed_time)}")

    print(f"\n[*] Total Cars IN: {cars_in} \tTotal Cars OUT: {cars_out}\
    \n[*] Video ended. \tFinising the program...\n")
    # Out of the loop, clean space
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fps.stop()
    fvs.stop()
