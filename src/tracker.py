import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

landmark_style = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
connection_style = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)

def start_tracking():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("NeonOptics Laboratuvarı Başladı. Çıkmak için 'q' tuşuna basın.")
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        black_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    black_canvas, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style
                )
                
        cv2.imshow("NeonOptics Lab", black_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_tracking()