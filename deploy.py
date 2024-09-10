from flask import Flask,render_template,Response
import cv2
import keras
import tensorflow
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.python.keras.layers import Dense

model = load_model('CNN_Model_acc_75.h5')

app=Flask(__name__)

img_shape = 48
# Define emotion labels
emotion_labels = ['angry',  'fear', 'happy', 'neutral','sad', 'surprise']
# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start capturing video
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

            #     # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
            #     # Rotate the frame 90 degrees left (experiment with other rotations if needed)
            #     # roi_color = cv2.rotate(roi_color, cv2.ROTATE_90_COUNTERCLOCKWISE)

                facess = face_cascade.detectMultiScale(roi_gray)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex,ey,ew,eh) in facess:
                        face_roi = roi_color[ey:ey+eh, ex:ex+ew]

                        # Predict emotions using the pre-trained model
                        final_image = cv2.resize(face_roi, (img_shape , img_shape))
                        final_image = np.expand_dims(final_image, axis=0)
                        final_image = final_image/ float(img_shape)
                        predictions = model.predict(final_image)
                        emotion = emotion_labels[np.argmax(predictions[0])]

            #             # Draw rectangle around face and label with predicted emotion
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # frame = gray_frame
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
