import cv2
import matplotlib.pyplot as pit
import numpy as np
from keras.applications import inception_v3

model = inception_v3.InceptionV3(weights='imagenet')
model.summary()

camera = cv2.VideoCapture(2)

rectangle = [(300, 75), (650, 425)]


def rescale_camera(frame):
    camera_height = 640
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)
    return cv2.resize(frame, (res, camera_height))


def draw_rectangle(frame):
    cv2.rectangle(
        frame, (rectangle[0][0], rectangle[0][1]), (rectangle[1][0], rectangle[1][1]), (240, 100, 0), 2)


def predict(area):
    captured_ares = np.array(
        [cv2.cvtColor(area_to_predict, cv2.COLOR_BGR2RGB)])
    return model.predict(captured_ares)


def draw_accuracy_labels(labels, frame, origin):
    starting_point = origin
    for i, label in enumerate(labels):
        accuracy = int(label[2]*100)
        if accuracy > 45:
            accuracy_label = "{} {} %".format(label[1], accuracy)
            cv2.putText(frame, accuracy_label,
                        (starting_point[0], starting_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 240, 150), 2)
            starting_point[1] += 30


def preprocess_area(area):
    area_to_predict = area[rectangle[0][1] +
                           2:rectangle[1][1]-2, rectangle[0][0]+2:rectangle[1][0]-2]
    rgb_area = cv2.cvtColor(area_to_predict, cv2.COLOR_BGR2RGB)
    resized_area = cv2.resize(rgb_area, (299, 299))
    area_to_predict = inception_v3.preprocess_input(resized_area)
    return area_to_predict


while(True):
    _, frame = camera.read()

    scaled_frame = rescale_camera(frame)

    draw_rectangle(scaled_frame)

    area_to_predict = preprocess_area(scaled_frame)

    predictions = predict(area_to_predict)
    labels = inception_v3.decode_predictions(predictions, top=3)[0]

    draw_accuracy_labels(labels, scaled_frame, [70, 170])

    cv2.imshow("Real Time object detection", scaled_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
