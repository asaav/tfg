import cv2
import numpy as np


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def print_stats(img, w, h, start):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    stats = "FPS: {0}\nResolution: {1}x{2}".format(fps, w, h)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(stats.split('\n')):
        cv2.putText(img, line, (10, 20+15*i), font, 0.8, (255, 255, 255))

    return img


def draw_contours(img, thresh):
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > 100]
    roundness = []
    for c in contours:
        # Calcular la circularidad de cada contorno mediante la formula C = (4*pi*area)/(perímetro^2)
        roundness.append(4*np.pi*cv2.contourArea(c)/cv2.arcLength(c, True)**2)

    for index, c in enumerate(contours):
        if index == np.argmax(roundness) and np.max(roundness) > 0.45:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img
