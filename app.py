from main import EmotionDetector
import streamlit as st

def main():
    onnx_model_path = 'emotion-ferplus-8.onnx'
    caffe_model_path = 'RFB-320/RFB-320.caffemodel'
    caffe_proto_path = 'RFB-320/RFB-320.prototxt'

    emotion_detector = EmotionDetector(onnx_model_path, caffe_model_path, caffe_proto_path)

 

    start_button = st.button("Start Video Streaming")
    stop_button = st.button("Stop Video Streaming")
    if start_button:
       count = emotion_detector.detect_emotions()

    if stop_button:
        emotion_detector.should_break()
        st.write("Video Streaming Stopped")
    


if __name__ == "__main__":
    main()
