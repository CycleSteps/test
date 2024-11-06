import pickle
import numpy as np
import face_recognition
import os
import cv2
from cryptography.fernet import Fernet

# Generate and save the encryption key
key = Fernet.generate_key()
cipher = Fernet(key)
with open('encryption_key.key', 'wb') as key_file:
    key_file.write(key)

# Path to the directory containing images
path = 'Images_Attendance'
images = []
classNames = []

# List of image files in the directory
myList = os.listdir(path)
print(f"Found images: {myList}")

# Load known images and their class names
for cl in myList:
    if cl.endswith(('jpg', 'jpeg', 'png')):  # Ensure only image files are read
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
print(f"Class names: {classNames}")

# Function to find encodings of known images
def findEncodings(images):
    encodeList = []  # Initialize the list inside the function
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure an encoding is present
            encodeList.append(encode[0])
    return encodeList

# Find encodings
encodeListKnown = findEncodings(images)

# Serialize and encrypt the encodings
serialized_encodings = pickle.dumps(encodeListKnown)
encrypted_encodings = cipher.encrypt(serialized_encodings)

# Save the encrypted encodings to 'encodings_list_encrypted.pkl'
with open('encodings_list_encrypted.pkl', 'wb') as file:
    file.write(encrypted_encodings)
print("Encrypted encodings saved to 'encodings_list_encrypted.pkl'.")

# Serialize and encrypt the class names
serialized_class_names = pickle.dumps(classNames)
encrypted_class_names = cipher.encrypt(serialized_class_names)

# Save the encrypted class names to 'class_names_encrypted.pkl'
with open('class_names_encrypted.pkl', 'wb') as file:
    file.write(encrypted_class_names)
print("Encrypted class names saved to 'class_names_encrypted.pkl'.")
