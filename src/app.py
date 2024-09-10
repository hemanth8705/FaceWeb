from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)

# Set the upload folder for videos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for both images and videos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi', 'mkv', 'webm'}

def allowed_file(filename):
    """Check if the file has an allowed video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the pre-trained model
model = load_model('CNN_Model_acc_75.h5')

img_shape = 48
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("static/demo_3x.mp4")

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        face_roi = cv2.resize(roi_color, (img_shape, img_shape))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / float(img_shape)
        predictions = model.predict(face_roi)
        emotion = ""
        emotion = emotion_labels[np.argmax(predictions[0])]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

    return frame

def generate_frames(source = "default_video"):
    global cap
    if source == 'camera':
        print("camera accessed")
        cap = cv2.VideoCapture(0)
    elif source == 'default_video':
        print("default_video accessed")
        cap = cv2.VideoCapture("static/demo_3x.mp4")
    elif source == 'video_uploaded':
        print(f"uploaded video  accessed uploads/{filename}")
        cap = cv2.VideoCapture("uploads/"+filename)
    else:
        print("default_video accessed")
        cap = cv2.VideoCapture("static/demo_3x.mp4")

    while True:
        success, frame = cap.read()
        if not success:
            print("frame is not success")
            break
        else:
            if source == 'image_uploaded':
                print(f"uploaded image  accessed uploads/{filename}")
                frame = cv2.imread("uploads/"+filename)
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if source == 'image_uploaded':
            break
    # cap.release()

@app.route('/handle_media_upload', methods=['POST'])
def handle_media_upload():
    """Handle video file upload."""
    # Check if the 'video_file' part exists in the request
    if 'media_file' not in request.files:
        return "No video file part", 400
    
    file = request.files['media_file']
    
    # If no file is selected, return an error
    if file.filename == '':
        return "No selected video file", 400
    
    # Check if the file is a valid image or video file
    if file and allowed_file(file.filename):
        # Get the file extension
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        global source , filename
        # Check if the file is an image or a video based on the extension
        if file_ext in {'png', 'jpg', 'jpeg', 'gif'}:
            filename = f"uploaded_image.{file_ext}"
            source = "image_uploaded"
        elif file_ext in {'mp4', 'mov', 'avi', 'mkv', 'webm'}:
            filename = f"uploaded_video.{file_ext}"
            source = "video_uploaded"
        # Secure the file name and save it to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"rendering file : {filename}")
        # Redirect or respond after successfully uploading
        return render_template('index.html')
    
    return "Invalid video file format", 400


@app.route('/handle_camera' , methods = ['POST',"GET"])
def handle_camera():
    global source 
    source = "camera"
    print("camera started")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/'  , methods = ['POST',"GET"])
def index():
    # cap.release()
    return render_template('index.html')

if __name__ == "__main__":
    source = "default_video"
    file_name = ""
    app.run(debug=True)
    # app.run(host ='0.0.0.0' , port = 8080)