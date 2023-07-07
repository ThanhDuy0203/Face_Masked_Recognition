from keras.models import load_model
import numpy as np 
import cv2

class detector_face_occlusion():
    def __init__(self):
        # network architecture
        prototxt_path = "face_detector/deploy.prototxt"
        # weights of the network
        caffemodel_path = "face_detector/weights.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def detect_face(self,image):
        (h, w) = image.shape[:2]
        # prepare the image to enter the model
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # input the image to the model
        self.detector.setInput(blob)
        # propagate the image forward of the model
        detections = self.detector.forward()
        """"
        detections, has 4 columns that are:
        0 column -->
        1st column -->
        2nd column --> number of detections made by default 200
        3th column --> has 7 sub_columns that are: 
             4.0 -->
             4.1 -->
             4.2 --> confidence
             4.3 --> x0
             4.4 --> y0
             4.5 --> x1
             4.6 --> y1
        """
        # Check the confidence of the 200's predictions
        list_box = []
        for i in range(0, detections.shape[2]):
            # box --> array[x0,y0,x1,y1]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # confidence range --> [0-1]
            confidence = detections[0,0,i,2]
            if confidence >=0.6:
                if list_box == []:
                    list_box = np.expand_dims(box,axis=0)
                else:
                    list_box = np.vstack((list_box,box))
        return list_box

