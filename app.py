import cv2
import streamlit as st
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    # Make a copy of the input frame to preserve the original
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Create a 4D blob from the image, required as input for the DNN model
    # Normalization: Subtract mean values (104, 117, 123) for RGB channels
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Set the blob as input to the network and perform a forward pass to get detections
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []  # Initialize a list to store the bounding boxes of detected faces
    for i in range(detections.shape[2]):
        # Extract confidence (probability) of the detection
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Calculate coordinates of the bounding box
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])  # Append the box to the list

            # Draw a rectangle around the detected face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes  # Return the processed frame and face bounding boxes

def process_image(image_path, faceNet, ageNet, genderNet):
    # Mean values for normalizing the input for age and gender prediction models
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Labels for age groups and genders
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load the input image
    image = cv2.imread(image_path)
    padding = 20  # Padding to include some area outside the detected face
    resultImg, faceBoxes = highlightFace(faceNet, image)  # Detect faces in the image
    if not faceBoxes:
        st.write("No face detected")  # Display message if no face is found
        return

    for faceBox in faceBoxes:
        # Extract the region of interest (ROI) for each detected face with padding
        face = image[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, image.shape[0] - 1), max(0, faceBox[0] - padding)
                     :min(faceBox[2] + padding, image.shape[1] - 1)]

        # Convert the face ROI into a blob, suitable for the prediction model
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender using the gender model
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # Get the gender label with the highest confidence

        # Predict age using the age model
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]  # Get the age group label with the highest confidence

        # Annotate the result image with gender and age predictions
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Convert the result image to RGB format for display in Streamlit
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    st.image(resultImg, channels="RGB", caption="Detected Image")  # Display the image in the app




def process_image_from_array(image, faceNet, ageNet, genderNet):
    # Mean values for normalizing input for the age and gender prediction models
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Labels for age groups and genders
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    padding = 20 # Padding to include some margin outside the detected face
    resultImg, faceBoxes = highlightFace(faceNet, image)
    if not faceBoxes:
        st.write("No face detected")  # If no face is detected, notify the user
        return

    for faceBox in faceBoxes:
        # Extract the region of interest (ROI) for the detected face, with padding
        face = image[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, image.shape[0] - 1), max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, image.shape[1] - 1)]
        
        # Convert the extracted face region into a blob, suitable for DNN models
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender using the gender detection model
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()] # Get the gender with the highest confidence

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Annotate the output image with the predicted gender and age
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    
    # Convert the BGR image to RGB format for compatibility with Streamlit display
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    # Display the processed image with detected faces and annotations
    st.image(resultImg, channels="RGB", caption="Processed Image")




def main():
    # Set the title for the Streamlit app
    st.title("Age and Gender Detection")

    # Define file paths for the models
    # Face detection model (architecture and weights)
    faceProto = "opencv_face_detector.pbtxt"  # Text description of the face detection model
    faceModel = "opencv_face_detector_uint8.pb"  # Pre-trained weights for face detection

    # Age detection model (architecture and weights)
    ageProto = "age_deploy.prototxt"  # Text description of the age detection model
    ageModel = "age_net.caffemodel"  # Pre-trained weights for age detection

    # Gender detection model (architecture and weights)
    genderProto = "gender_deploy.prototxt"  # Text description of the gender detection model
    genderModel = "gender_net.caffemodel"  # Pre-trained weights for gender detection

    # Load the models using OpenCV's DNN module
    faceNet = cv2.dnn.readNet(faceModel, faceProto)  # Load face detection model
    ageNet = cv2.dnn.readNet(ageModel, ageProto)  # Load age prediction model
    genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Load gender prediction model

    # Streamlit dropdown menu to select the mode of operation
    option = st.selectbox("Choose an option:", ["Upload Image", "Live Camera", "Capture Photo"])

    # Option 1: Upload an image file
    if option == "Upload Image":
        # File uploader widget to allow users to upload an image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save the uploaded file locally
            image_path = uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Process the uploaded image for face detection, age, and gender prediction
            process_image(image_path, faceNet, ageNet, genderNet)

    # Option 2: Live camera feed (not functional in this implementation due to server constraints)
    elif option == "Live Camera":
        # Inform the user that the live camera option is unavailable
        st.write("Our server has no camera so you can't use this option yet. Use 'Capture Photo' option to take live snapshots from your camera.")

    # Option 3: Capture a photo using the camera
    elif option == "Capture Photo":
        # Camera input widget to allow users to take a photo
        img_file_buffer = st.camera_input("Take a photo using your camera")
        if img_file_buffer is not None:
            # Convert the captured photo into a numpy array
            image = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
            # Decode the numpy array to an OpenCV image
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # Process the captured image for face detection, age, and gender prediction
            process_image_from_array(image, faceNet, ageNet, genderNet)



if __name__ == "__main__":
    main()
