import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
import threading

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, max_num_hands=1)

labels_dict = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'j': 'j',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'q': 'q',
    'r': 'r',
    's': 's',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'x': 'x',
    'y': 'y',
    'z': 'z',
    'hello': 'Hello',
    'space': 'Espaco',
}

# label_var = None

def inference_classifier():
    lastPredictedCharacter = ''
    words = ''
    contador = 0
    # global label_var
    
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])

                predicted_character = labels_dict[prediction[0]]

                if contador >= 600:
                    if predicted_character == 'Espaco':
                        words = words + ' '
                    else:
                        words = words + lastPredictedCharacter

                    # Update the label text in the tkinter window
                    label_var.set(words)
                    contador = 0
                elif predicted_character != lastPredictedCharacter:
                    contador = 0
                else:
                    contador += 35
                    cv2.rectangle(frame, (0, 40), (contador, 40), (124, 252, 0), 30)

                lastPredictedCharacter = predicted_character

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

def display_tkinter():
    global label_var
    
    # Create a new window
    window = tk.Tk()

    # Set the window title
    window.title("Output sign recognized")

    # Set the window Width x Height
    window.geometry("600x400")

    # Create a StringVar to store the label text
    label_var = tk.StringVar()
    label_var.set("Waiting for CV2 processing...")

    # Create a label widget to display the text
    label = tk.Label(window, textvariable=label_var, font=("Arial", 25), anchor='w')
    label.pack(fill='both')

    # Run the tkinter event loop
    window.mainloop()


# Create and start the threads for inference_classifier cv2 and tkinter windows
tkinter_thread = threading.Thread(target=display_tkinter)
tkinter_thread.start()

cv2_thread = threading.Thread(target=inference_classifier)
cv2_thread.start()
