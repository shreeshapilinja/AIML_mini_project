import cv2
import os
import time
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import face_recognition
from fer import FER
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import pyttsx3
import pygame
import random
import webbrowser

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


# face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# dataset for face recognition
dataset_path = "datasets"

# hand detection
detector =  HandDetector(detectionCon=0.8,maxHands=2)

#fer emotion detection
emo_detector = FER()  #emo_detector = FER(mtcnn=True)

# music play
play_music = None
pygame.init()


cap = cv2.VideoCapture(0)
fps_start_time = time.time()
fps = 0

# Load known face encodings and names for face_recognition
known_face_encodings, known_face_names = known_face_recognition_load(dataset_path)


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
                text_to_speech_male(f"playing music for no people or many people who have emotions as {emotion}")
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

def face_recognition_func(img):
    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    
    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.6)  # Check if the face matches any known faces : tolerance - How much distance between faces to consider it a match.
        faceDis = face_recognition.face_distance(known_face_encodings, face_encoding)
        matchIndex = np.argmin(faceDis)   
        name = "unknown" 
        if True in matches and matches[matchIndex]:
            #first_match_index = matches.index(True)
            name = known_face_names[matchIndex]
        if name not in names:
            names.append(name)

    if names:
        #print(str(names))
        return names[0]
    else:
        #print("unknown")
        return "unknown"

def analyze_emotion_fer_fuc(img):
    emotions = []
    probabilities = []
    try:
        result = emo_detector.detect_emotions(img)
        for face in result:
            emotion = max(face['emotions'], key=face['emotions'].get)
            probability = round(face['emotions'][emotion]*100,2)
            emotions.append(emotion)
            probabilities.append(str(probability))
    except:
        try:
            result = DeepFace.analyze(img,actions=("emotion",),enforce_detection=False,detector_backend='mediapipe',align=True,silent=False)
            for face in result:
                emotion = face['dominant_emotion']
                probability = round(face['emotion'][emotion],2)
                emotions.append(emotion)
                probabilities.append(str(probability))
        except:
            print("Face partially visible for emotion detection")
    if len(emotions):
        #print(str(emotions))
        return str(emotions[0]),probabilities[0]
    else:
        return 'Neutral','15'
    
def analyze_emotion_deep_fuc(img):
    emotions = []
    probabilities = []
    try:
        result = DeepFace.analyze(img,actions=("emotion",),enforce_detection=False,detector_backend='mediapipe',align=True,silent=False)
        for face in result:
            emotion = face['dominant_emotion']
            probability = round(face['emotion'][emotion],2)
            emotions.append(emotion)
            probabilities.append(str(probability))
    except:
        try:
            result = emo_detector.detect_emotions(img)
            for face in result:
                emotion = max(face['emotions'], key=face['emotions'].get)
                probability = round(face['emotions'][emotion]*100,2)
                emotions.append(emotion)
                probabilities.append(str(probability))
        except:
            print("Face partially visible for emotion detection")
    if len(emotions):
        #print(str(emotions))
        return str(emotions[0]),probabilities[0]
    else:
        return 'Neutral','15'


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        #frame = cv2.flip(frame, 1) # Flip the image horizontally
        if ret == False:
            break
        
        names = []
        emotions = []
        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(temp_frame)
        hands,frame = detector.findHands(frame,flipType=True)  # draw=False for no draw,by default True and flipType=False for flip,  by default True
        
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                
                x_start, y_start = int(box.xmin * frame.shape[1]), int(box.ymin * frame.shape[0])
                x_end, y_end = int((box.xmin + box.width) * frame.shape[1]), int((box.ymin + box.height) * frame.shape[0])
                
                annotated_image = cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                
                score = "Face: " + str(detection.score)[3:5] + "%"
                cv2.putText(annotated_image, score, (x_start + 2, y_start + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                # Crop the detected face region
                cropped_face = temp_frame[y_start-85 : y_end+35, x_start-50 : x_end+50]
                try:
                    name = face_recognition_func(cropped_face)
                    cv2.putText(annotated_image, name, (x_start+10 , y_start - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    names.append(name)
                except:
                    print("Name Error")
                    
                try:
                    emotion,probability = analyze_emotion_deep_fuc(cropped_face)
                    #emotion,probability = analyze_emotion_fer_fuc(cropped_face)
                    text = str(emotion) + " " + str(probability) + "%"
                    cv2.putText(annotated_image, text, (x_start + 10, y_start - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    #if emotion != 'Neutral':
                    emotions.append(emotion.lower())
                except:
                    print("Emotion Error")
                    
            try:
                if hands:
                    play_music_func(hands,names,emotions)
            except:
                print("Error in music playing")
                    
        # Calculate FPS
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face & Emotion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
