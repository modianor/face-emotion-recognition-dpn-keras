import time

import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.engine.saving import load_model, model_from_json

import detect_face
from constants import EMOTIONS, SIZE_FACE

npy = './npy'
feelings_faces = []
# cv2.namedWindow('emotion',cv2.WINDOW_NORMAL)

for index, emotion in enumerate(EMOTIONS):
    em = cv2.imread('./emojis/' + emotion + '.png', -1)
    feelings_faces.append(em)
    # cv2.imshow('emotion', em)
    # cv2.waitKey(0)



with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        # margin = 44
        frame_interval = 3
        # batch_size = 1000
        # image_size = 182
        # input_image_size = 160

        # video_capture = cv2.VideoCapture(input_video)
        video_capture = cv2.VideoCapture(0)
        c = 0
        with open('model/fer2013/model.json', 'r') as file:
            model_json = file.read()

        model: Sequential = model_from_json(model_json)
        model.load_weights('model/fer2013/model.h5')
        font = cv2.FONT_HERSHEY_SIMPLEX

        print('Start Recognition')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            min_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

            curTime = time.time() + 1  # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                # if frame.ndim == 2:
                #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        face = frame[bb[i][0]:bb[i][2], bb[i][1]:bb[i][3]]

                        # if face is None :
                        #     continue

                        if face.shape is None or face.shape[0] == 0:
                            continue

                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        min_face = cv2.resize(gray_face, (112, 112),
                                              interpolation=cv2.INTER_CUBIC) / 255.
                        print(min_face.shape)
                        min_face = min_face.reshape((1, 112, 112, 1))
                        result = model.predict(min_face, batch_size=1, verbose=1)
                        print(type(result), result)
                        # Draw face in frame
                        # for (x,y,w,h) in faces:
                        #   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        print(result[0, :])
                        # Write results in frame
                        if result is not None:
                            for index, emotion in enumerate(EMOTIONS):
                                cv2.putText(frame, emotion, (10, index * 20 + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                                cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                                              int(result[0, index:index + 1] * 100),
                                                                              (index + 1) * 20 + 4), (255, 0, 0), -1)

                            face_image = feelings_faces[np.argmax(result[0, :])]

                            # Ugly transparent fix
                            for cc in range(0, 3):
                                # print(frame.shape, face_image.shape)
                                frame[100:220, 150:270, cc] = face_image[:, :, cc] * (
                                        face_image[:, :, 3] / 255.0) + frame[
                                                                       100:220,
                                                                       150:270,
                                                                       cc] * (
                                                                      1.0 - face_image[:, :, 3] / 255.0)

                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face
                else:
                    print('Alignment Failure')
            # c+=1
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
