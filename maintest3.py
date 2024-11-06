import cv2
import numpy as np
import face_recognition
import pickle
from cryptography.fernet import Fernet

# Load known face encodings and their names
def image_fetch():
    global encodeListKnown, classNames
    # Load the encryption key
    try:
        with open('encryption_key.key', 'rb') as key_file:
            key = key_file.read()
    except FileNotFoundError:
        print("Error: 'encryption_key.key' not found.")
        exit()

    cipher = Fernet(key)

    # Load and decrypt the encodings list
    try:
        with open('encodings_list_encrypted.pkl', 'rb') as file:
            encrypted_encode_list = file.read()
        decrypted_encode_list = cipher.decrypt(encrypted_encode_list)
        loaded_encode_list = pickle.loads(decrypted_encode_list)
        print("Encodings loaded successfully.")
    except FileNotFoundError:
        print("Error: 'encodings_list_encrypted.pkl' not found.")
        exit()
    except Exception as e:
        print(f"Error decrypting or loading encodings: {e}")
        exit()

    # Load and decrypt the class names
    try:
        with open('class_names_encrypted.pkl', 'rb') as file:
            encrypted_class_names = file.read()
        decrypted_class_names = cipher.decrypt(encrypted_class_names)
        classNames = pickle.loads(decrypted_class_names)
        print("Class names loaded successfully.")
    except FileNotFoundError:
        print("Error: 'class_names_encrypted.pkl' not found.")
        exit()
    except Exception as e:
        print(f"Error decrypting or loading class names: {e}")
        exit()

    encodeListKnown = loaded_encode_list


# Function to check if a person in the input image matches known encodings
def check_person(input_img_rgb):
    faces_in_input = face_recognition.face_locations(input_img_rgb)
    encodes_in_input = face_recognition.face_encodings(input_img_rgb, faces_in_input)

    # Variable to check if any match was found
    match_found = False

    for encodeFace, faceLoc in zip(encodes_in_input, faces_in_input):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Get the index of the best match
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f'Match found: {name}')
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(input_img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(input_img_rgb, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            match_found = True  # Mark that a match was found

    if not match_found:
        print('No match found')

    # Display the input image with the result
    cv2.imshow('Result', cv2.cvtColor(input_img_rgb, cv2.COLOR_RGB2BGR))

# Initialize the face detection model and capture video
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    image_fetch()  # Load encodings once before starting detection

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            print("Face detected!")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            check_person(rgb_frame)  # Check the captured frame for known faces
            cv2.waitKey(0)  # Pause to view the results
            cv2.destroyAllWindows()
            break

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
