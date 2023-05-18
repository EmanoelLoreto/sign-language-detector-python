import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, max_num_hands=2)

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
    'w': 'w',
    'x': 'x',
    'y': 'y',
    'z': 'z',
    'hello': 'Hello',
}

lastPredictedCharacters = ['', '']

while True:
    data_aux = [[], []]
    x_ = [[], []]
    y_ = [[], []]

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

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y

                x_[i].append(x)
                y_[i].append(y)

                data_aux[i].append(x)
                data_aux[i].append(y)

            x1 = int(min(x_[i]) * W) - 10
            y1 = int(min(y_[i]) * H) - 10

            x2 = int(max(x_[i]) * W) + 10
            y2 = int(max(y_[i]) * H) + 10

            if len(data_aux[i]) == 42:
                min_x = min(x_[i])
                min_y = min(y_[i])
                normalized_data = [(x - min_x, y - min_y) for x, y in zip(x_[i], y_[i])]
                reshaped_data = np.asarray(normalized_data).reshape(1, -1)
                prediction = model.predict(reshaped_data)
                predicted_character = labels_dict[prediction[0]]

                if (predicted_character != lastPredictedCharacters[i]):
                    print(f"Hand {i + 1}: {predicted_character}")

                lastPredictedCharacters[i] = predicted_character

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
