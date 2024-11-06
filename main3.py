import cv2
import mediapipe as mp
import face_recognition
import os

# Define the path where your images are stored
path = '.venv/New_Images'  # Set the correct path to your images directory

# Initialize lists to store images and names
images = []
classNames = []

# List all files in the directory
myList = os.listdir(path)
print(f"Found images: {myList}")

# Load known images and their class names (names are based on image file names)
for cl in myList:
    if cl.endswith(('jpg', 'jpeg', 'png')):  # Ensure only image files are read
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])  # Use the image name (without extension) as the label
print(f"Class names: {classNames}")

# Function to find face encodings for a list of images
def findEncodings(images):
    encodeList = []  # Initialize the list inside the function
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for face_recognition
        encode = face_recognition.face_encodings(img)  # Get the face encodings
        if encode:  # Ensure an encoding is present
            encodeList.append(encode[0])  # Append the encoding (first face if multiple detected)
    return encodeList

# Find encodings for the known faces
encodeListKnown = findEncodings(images)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up the Face Detection object
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open the camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face detection
    results = face_detection.process(rgb_frame)

    # Check if any faces are detected
    if results.detections:
        for detection in results.detections:
            # Get the bounding box of the face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face from the frame
            face = frame[y:y + h, x:x + w]

            # Check if the face is empty (i.e., no valid face was detected)
            if face.size == 0:
                continue  # Skip to the next detected face (if any)

            # Convert the face to RGB (face_recognition works with RGB images)
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Find face encodings in the detected face
            face_encoding = face_recognition.face_encodings(rgb_face)

            if face_encoding:
                # Compare with known faces
                matches = face_recognition.compare_faces(encodeListKnown, face_encoding[0])
                name = "Unknown"

                # If a match is found, set the name of the person
                if True in matches:
                    first_match_index = matches.index(True)
                    name = classNames[first_match_index]

                # Draw bounding box and name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
