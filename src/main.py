import struct

import cv2
import numpy as np
import socket


THRESHOLD = 28
THRESHOLD_RANGE = 40
SCALE = 1.0
REAL_EYE_DISTANCE = 6.5


def capture_photo():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv2.imshow('title', frame)
        key = cv2.waitKey(10)
        if key == 27:
            break
        elif key == 32:
            cv2.imwrite('photo1.png', frame)
        elif key != -1:
            print(key)

    vid.release()
    cv2.destroyAllWindows()
    
    
def detect_face(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for coord in coords:
            if coord[3] > biggest[3]:
                biggest = coord
    elif len(coords) == 1:
        biggest = coords[0]
    else:
        return None, None
    x, y, w, h = biggest
    return img[y:y + h, x:x + w], biggest


def detect_eyes(img, classifier):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray
    eyes = classifier.detectMultiScale(img_gray, 1.3, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    left_eye_coords = None
    right_eye = None
    right_eye_coords = None
    for x, y, w, h in eyes:
        if y > height/2:
            continue
        eye_center = x + w / 2  # get the eye center
        if eye_center < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            left_eye_coords = x, y, w, h
        else:
            right_eye = img[y:y + h, x:x + w]
            right_eye_coords = x, y, w, h
    return (left_eye, left_eye_coords), (right_eye, right_eye_coords)


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)return img

    return img, eyebrow_h


def blob_process(img, detector, begin_threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = None
    threshold = begin_threshold
    while not keypoints:
        _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=1)
        img = cv2.dilate(img, None, iterations=2)
        img = cv2.medianBlur(img, 3)
        keypoints = detector.detect(img)
        threshold += 2
        if threshold > begin_threshold + THRESHOLD_RANGE:
            break
    return keypoints, img


def main():
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector = cv2.SimpleBlobDetector_create(detector_params)

    face_cascade = cv2.CascadeClassifier('models/face_model.xml')
    eye_cascade = cv2.CascadeClassifier('models/eye_model.xml')

    scale = SCALE
    fx, fy, fz = 0.0, 0.0, 1.0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        vid = cv2.VideoCapture(0)
        while True:
            ret, img = vid.read()

            # img = cv2.medianBlur(img, 5)

            keypoints = []
            global_eye_coords = []

            face, face_coords = detect_face(img, face_cascade)
            if face is not None:
                for index, eye_and_coords in enumerate(detect_eyes(face, eye_cascade)):
                    eye, eye_coords = eye_and_coords
                    if eye is not None:
                        eye, eyebrow_height = cut_eyebrows(eye)
                        global_eye_coords.append(eye_coords)

                        eye_keypoints, edit_img = blob_process(eye, detector, THRESHOLD)
                        if eye_keypoints:
                            keypoint = eye_keypoints[0]
                            keypoint.pt = (
                                keypoint.pt[0] + face_coords[0] + eye_coords[0],
                                keypoint.pt[1] + face_coords[1] + eye_coords[1] + eyebrow_height
                            )
                            keypoints.append((keypoint, index))

            for eye_coord in global_eye_coords:
                x, y, w, h = eye_coord
                x += face_coords[0]
                y += face_coords[1]
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

            if keypoints:
                cv2.drawKeypoints(
                    img,
                    list(map(lambda kp: kp[0], keypoints)),
                    img,
                    (0, 0, 255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                indices_found = [False, False]
                for keypoint, index in keypoints:
                    x, y = keypoint.pt
                    if index == 0:
                        fx = (x - 300) / 300 * scale
                        fy = (y - 250) / 250 * scale
                    indices_found[index] = x, y

                if all(indices_found):
                    left_pos_x, left_pos_y = indices_found[0]
                    right_pos_x, right_pos_y = indices_found[1]
                    a = left_pos_x - right_pos_x
                    b = left_pos_y - right_pos_y
                    eye_distance = np.sqrt(a*a + b*b)
                    fz = REAL_EYE_DISTANCE / (2 * np.tan(eye_distance/100.0))
                    print('fz: ', fz)

                sock.sendto(struct.pack("fff", fx, fy, fz), ('127.0.0.1', 1351))

            cv2.imshow('img', img)

            key = cv2.waitKey(10)
            if key == 27:
                break
            elif key == ord('k'):
                scale += 0.1
                print('scale: ', scale)
            elif key == ord('j'):
                scale -= 0.1
                print('scale: ', scale)

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
