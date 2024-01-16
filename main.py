import numpy as np
import cv2
import time 
import math
import streamlit as st

class EmotionDetector:
    def __init__(self, onnx_model_path, caffe_model_path, caffe_proto_path):
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [
            [10.0, 16.0, 24.0], 
            [32.0, 48.0], 
            [64.0, 96.0], 
            [128.0, 192.0, 256.0]
        ]
        self.strides = [8.0, 16.0, 32.0, 64.0]
        self.threshold = 0.5
        self.model = cv2.dnn.readNetFromONNX("RFB-320/RFB-320.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(caffe_proto_path, caffe_model_path)
        self.input_size = [320, 240]
        self.width = self.input_size[0]
        self.height = self.input_size[1]
        self.priors = self.define_img_size(self.input_size)
        self.emotion_dict = {
            0: 'neutral', 
            1: 'smiling', 
            2: 'surprise', 
            3: 'sadness',
            4: 'anger', 
            5: 'disgust', 
            6: 'fear'
        }


    def define_img_size(self, image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(math.ceil(size / stride)) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)
        
        for _ in range(len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self.generate_priors(
            feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes
        )
        return priors
    
    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h
         
                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
        print("priors nums:{}".format(len(priors)))
        return np.clip(priors, 0.0, 1.0)
    
    def convert_locations_to_boxes(self, locations):
        if len(self.priors.shape) + 1 == len(locations.shape):
            self.priors = np.expand_dims(self.priors, 0)
        return np.concatenate([
            locations[..., :2] * self.center_variance * self.priors[..., 2:] + self.priors[..., :2],
            np.exp(locations[..., 2:] * self.size_variance) * self.priors[..., 2:]
        ], axis=len(locations.shape) - 1)
    
    @staticmethod
    def area_of(left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]
   
    @staticmethod
    def iou_of(boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = EmotionDetector.area_of(overlap_left_top, overlap_right_bottom)
        area0 = EmotionDetector.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = EmotionDetector.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)
    
    @staticmethod
    def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = EmotionDetector.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]
    
    def predict(
        self,
        width, 
        height, 
        confidences, 
        boxes, 
        prob_threshold, 
        iou_threshold=0.3, 
        top_k=-1
        ):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
              continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1
            )
            box_probs = EmotionDetector.hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return (
            picked_box_probs[:, :4].astype(np.int32), 
            np.array(picked_labels), 
            picked_box_probs[:, 4]
            )
    
    def convert_locations_to_boxes(self,locations, priors, center_variance,
                               size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)
    
    def center_form_to_corner_form(self,locations):
        return np.concatenate(
            [locations[..., :2] - locations[..., 2:] / 2,
            locations[..., :2] + locations[..., 2:] / 2], 
            len(locations.shape) - 1
            )
    def should_break(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def detect_emotions(self):
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        emotion_counts = {emotion: 0 for emotion in self.emotion_dict.values()}
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    img_ori = frame
                    rect = cv2.resize(img_ori, (self.width, self.height))
                    rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
                    self.net.setInput(cv2.dnn.blobFromImage(
                        rect, 1 / self.image_std, (self.width, self.height), 127)
                     )
                    start_time = time.time()
                    boxes, scores = self.net.forward(["boxes", "scores"])
                    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
                    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
                    boxes = self.convert_locations_to_boxes(
                        boxes, self.priors, self.center_variance, self.size_variance
                    )
                    boxes = self.center_form_to_corner_form(boxes)
                    boxes, labels, probs = self.predict(
                        img_ori.shape[1], 
                        img_ori.shape[0], 
                        scores, 
                        boxes, 
                        self.threshold
                    )
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for (x1, y1, x2, y2) in boxes:
                        w = x2 - x1
                        h = y2 - y1
                        cv2.rectangle(frame, (x1,y1), (x2, y2), (255,0,0), 2)
                        resize_frame = cv2.resize(
                            gray[y1:y1 + h, x1:x1 + w], (64, 64)
                        )
                        resize_frame = resize_frame.reshape(1, 1, 64, 64)
                        self.model.setInput(resize_frame)
                        output = self.model.forward()
                        end_time = time.time()
                        fps = 1 / (end_time - start_time)
                        # print(f"FPS: {fps:.1f}")
                        pred = self.emotion_dict[list(output[0]).index(max(output[0]))]
                        # print(pred)
                        emotion_counts[pred] += 1
                        # print(emotion_counts)
                        cv2.rectangle(
                            img_ori, 
                            (x1, y1), 
                            (x2, y2), 
                            (0, 255, 0), 
                            2,
                            lineType=cv2.LINE_AA
                        )
                        cv2.putText(
                            frame, 
                            pred, 
                            (x1, y1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 255, 0), 
                            2,
                            lineType=cv2.LINE_AA
                            )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
                    if self.should_break():
                        break
                else:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
        return emotion_counts


from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase

class EmotionVideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_dict.values()}

    def recv(self, frame):
        img_ori = frame.to_ndarray(format="bgr24")
        rect = cv2.resize(img_ori, (self.width, self.height))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        self.net.setInput(cv2.dnn.blobFromImage(
            rect, 1 / self.image_std, (self.width, self.height), 127)
        )
        start_time = time.time()
        boxes, scores = self.net.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = self.convert_locations_to_boxes(
            boxes, self.priors, self.center_variance, self.size_variance
        )
        boxes = self.center_form_to_corner_form(boxes)
        boxes, labels, probs = self.predict(
            img_ori.shape[1], 
            img_ori.shape[0], 
            scores, 
            boxes, 
            self.threshold
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(frame, (x1,y1), (x2, y2), (255,0,0), 2)
            resize_frame = cv2.resize(
                gray[y1:y1 + h, x1:x1 + w], (64, 64)
            )
            resize_frame = resize_frame.reshape(1, 1, 64, 64)
            self.model.setInput(resize_frame)
            output = self.model.forward()
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            pred = self.emotion_dict[list(output[0]).index(max(output[0]))]
            self.emotion_counts[pred] += 1
            cv2.rectangle(
                img_ori, 
                (x1, y1), 
                (x2, y2), 
                (0, 255, 0), 
                2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame, 
                pred, 
                (x1, y1), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2,
                lineType=cv2.LINE_AA
                )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

def detect_emotions(self):
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}), video_processor_factory=EmotionVideoTransformer)
    return self.emotion_counts

# onnx_model_path = 'emotion-ferplus-8.onnx'
# caffe_model_path = 'RFB-320/RFB-320.caffemodel'
# caffe_proto_path = 'RFB-320/RFB-320.prototxt'
# emotion_detector = EmotionDetector(onnx_model_path, caffe_model_path, caffe_proto_path)
# count = emotion_detector.detect_emotions()
# print(count)
# print(type(count))
