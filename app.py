import cv2
import streamlit as st
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def process_image(image_path, faceNet, ageNet, genderNet):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    image = cv2.imread(image_path)
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, image)
    if not faceBoxes:
        st.write("No face detected")
        return

    for faceBox in faceBoxes:
        face = image[max(0, faceBox[1] - padding):
                    min(faceBox[3] + padding, image.shape[0] - 1), max(0, faceBox[0] - padding)
                    :min(faceBox[2] + padding, image.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Convert the BGR image to RGB for displaying in Streamlit
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    st.image(resultImg, channels="RGB", caption="Detected Image")

def process_video(faceNet, ageNet, genderNet):
    padding = 20
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            st.write("Failed to grab frame.")
            break
        
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            st.write("No face detected")
            continue

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                        min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                        :min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderPreds[0].argmax()

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = agePreds[0].argmax()

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Convert the BGR image to RGB for displaying in Streamlit
        resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
        st.image(resultImg, channels="RGB", caption="Live Video Feed")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()

def process_image_from_array(image, faceNet, ageNet, genderNet):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, image)
    if not faceBoxes:
        st.write("No face detected")
        return

    for faceBox in faceBoxes:
        face = image[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, image.shape[0] - 1), max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, image.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # BGR to RGB conversion for Streamlit display
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    st.image(resultImg, channels="RGB", caption="Processed Image")


def capture_and_process_image(faceNet, ageNet, genderNet):
    padding = 20
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture image")
        return
    
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        st.write("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                    min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                    :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderPreds[0].argmax()

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = agePreds[0].argmax()

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Convert the BGR image to RGB for displaying in Streamlit
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    st.image(resultImg, channels="RGB", caption="Captured Image")

    camera.release()

def main():
    st.title("Age and Gender Detection")

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    option = st.selectbox("Choose an option:", ["Upload Image", "Live Camera", "Capture Photo"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_path = uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            process_image(image_path, faceNet, ageNet, genderNet)

    elif option == "Live Camera":
        st.write("Our server has no camera so you cant use this option yet. Use 'Capture Photo' option to take live snapshots from your camera.")

    elif option == "Capture Photo":
        img_file_buffer = st.camera_input("Take a photo using your camera")
        if img_file_buffer is not None:
            image = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            process_image_from_array(image, faceNet, ageNet, genderNet)


if __name__ == "__main__":
    main()
