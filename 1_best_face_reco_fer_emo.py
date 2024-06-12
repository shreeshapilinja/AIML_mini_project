import face_recognition
from fer import FER
import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import pyttsx3
import pygame
import random
import webbrowser

dataset_path = "datasets"
detector =  HandDetector(detectionCon=0.8,maxHands=2)
play_music = None
pygame.init()
#emo_detector = FER(mtcnn=True)
emo_detector = FER()


# For face_recognition module 
def known_face_recognition_load(directory="datasets"):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(directory):
        person_folder = os.path.join(directory, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                    # Load known face image
                    known_person_image = face_recognition.load_image_file(os.path.join(person_folder, filename))
                    
                    # Extract face encoding
                    face_encodings = face_recognition.face_encodings(known_person_image)
                    
                    # Check if any face encodings were found
                    if face_encodings:
                        known_person_encoding = face_encodings[0]
                    else:
                        # Handle the case where no face encoding is found
                        print(f"No face encoding found for {filename}")
                        continue  # Skip this iteration and move to the next image

                    # Append the encoding and name to the lists
                    known_face_encodings.append(known_person_encoding)
                    known_face_names.append(person_name)

    return known_face_encodings, known_face_names
    
# Load known face encodings and names for face_recognition
face_known_face_encodings, face_known_face_names = known_face_recognition_load(dataset_path)

def fer_analyze_emotion(img):
    emotions = []
    result = emo_detector.detect_emotions(img)
    for face in result:
        box = face['box']
        emotion = max(face['emotions'], key=face['emotions'].get)
        probability = face['emotions'][emotion]
        # Draw bounding box
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        # Display emotion text
        cv2.putText(img, f"{emotion}: {probability:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        emotions.append(emotion)
    emotions = list(set(emotions))
    return img,emotions

def text_to_speech_male(text, rate=200):  # Adjust the rate value as needed
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # 0 for male, 1 for female
    engine.setProperty('rate', rate)  # 200 words per minute by default
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def play_music_func(hands,names,emotions):
    global play_music
    print(len(names),str(names),str(emotions))
    if len(hands) == 1:
        hand = hands[0]
        hand_type = hand["type"]
        #print(hand_type)
        if hand_type == "Right" and (play_music==False or play_music==None):
            if len(names) == 1:
                name = names[0]
                if len(emotions) == 0:
                    emotions.append('neutral')
                if name == "unknown":
                    print(f"playing music for new person who is {emotions[0]}")
                    text_to_speech_male(f"playing music for new person who is {emotions[0]}")
                    # Load the music file
                    pygame.mixer.music.load(f"songs/unknown/{emotions[0]}/{random.randint(1, 3)}.mp3") # (both included)
                    # Play the music
                    pygame.mixer.music.play()
                else:
                    print(f"playing music for {name} who is {emotions[0]}")
                    text_to_speech_male(f"playing music for {name} who is {emotions[0]}")
                    pygame.mixer.music.load(f"songs/{name}/{emotions[0]}/{random.randint(1, 3)}.mp3")
                    pygame.mixer.music.play()
            else:
                emotion = ' '.join(emotions)
                print(f"playing music for no people or many people who have emotions as {emotion}")
                text_to_speech_male(f"playing music for many people who have emotions as {emotion}")
                pygame.mixer.music.load(f"songs/many/{random.randint(1, 5)}.mp3")
                pygame.mixer.music.play()
            play_music = True
        elif hand_type == "Left" and play_music == True:
            print("Stoping the music")
            pygame.mixer.music.stop()
            text_to_speech_male("Music Stopped")
            play_music = False
    elif len(hands) == 2 and (play_music==False or play_music==None):
        if len(emotions) == 1:
            webbrowser.open(f"https://pixabay.com/music/search/mood/{emotions[0]}/")
            text_to_speech_male(f"opening music for {emotions[0]}")
            play_music = True
        elif len(emotions) == 2:
            webbrowser.open(f"https://pixabay.com/music/search/mood/{emotions[0]}/?mood={emotions[1]}")
            text_to_speech_male(f"opening music for {emotions[0]} and {emotions[1]}")
            play_music = True
        elif len(emotions) == 3:
            webbrowser.open(f"https://pixabay.com/music/search/mood/{emotions[0]}/?mood={emotions[1]}&mood={emotions[2]}")
            text_to_speech_male(f"opening music for {emotions[0]}, {emotions[1]} and {emotions[2]}")
            play_music = True
        else:
            text_to_speech_male("Please raise only one hand at a time and there are many emotions")
            print("Raised 2 or many hands at a time and there are many emotions")

def face_recognition_library(img):
    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(img)
    #face_locations = face_recognition.face_locations(img,model="hog")
    face_encodings = face_recognition.face_encodings(img, face_locations)

    img,emotions = fer_analyze_emotion(img)
    names = []
    
    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(face_known_face_encodings, face_encoding)   # Check if the face matches any known faces
        faceDis = face_recognition.face_distance(face_known_face_encodings, face_encoding)
        matchIndex = np.argmin(faceDis)
        
        #matches = face_recognition.compare_faces(face_known_face_encodings, face_encoding,tolerance=0.4)
        name = "unknown"
        if True in matches and matches[matchIndex]:
            first_match_index = matches.index(True)
            name = face_known_face_names[first_match_index]
            if name in names:
                name = "unknown"
        names.append(name.lower())
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)     # Draw a box around the face and label with the name
        cv2.putText(img, name.title(), (left+2 , top+28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    hands,img = detector.findHands(img)  # with drawing and flipType=True  by default True
    if hands:
        play_music_func(hands,names,emotions)
    return img
    
    
# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    result = face_recognition_library(frame)
        
    # Display the resulting frame
    cv2.imshow("Face&Emotion", result)
    
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()