from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
import os
import io
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
from .forms import ImageUploadForm
from django.conf import settings

BASE_DIR = settings.BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_path = os.path.join(MODEL_DIR, "mask_detection.h5")
prototxt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
caffemodel_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

face_mask_model = load_model(model_path)
face_net = cv.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

LABELS = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}

def index(request):
    """ Render the home page. """
    return render(request, 'index.html')

def mask_detection(image):
    """
    Perform face mask detection on the given image.
    :param image: OpenCV image (numpy.ndarray)
    :return: Processed OpenCV image
    """
    
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    (h, w) = image.shape[:2]

    # Detect faces
    blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = tf.expand_dims(face, axis=0)
            face = preprocess_input(face)

            prediction = face_mask_model.predict(face)
            label_idx = np.argmax(prediction, axis=1)[0]
            label = LABELS[label_idx]

            color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)

            cv.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv.putText(image, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

def process_image(request):
    """
    Handle image upload and process mask detection.
    """
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']

            image = Image.open(uploaded_image)

            processed_image = mask_detection(image)

            image_io = io.BytesIO()
            processed_image = cv.cvtColor(processed_image, cv.COLOR_BGR2RGB)
            processed_image = Image.fromarray(processed_image)
            processed_image.save(image_io, format='PNG')
            image_io.seek(0)

            return HttpResponse(image_io.read(), content_type="image/png")

    else:
        form = ImageUploadForm()

    return render(request, "upload.html", {"form": form})

def live_feed(request):
    """ Return a live video stream with mask detection applied. """
    return StreamingHttpResponse(video_stream(), content_type="multipart/x-mixed-replace; boundary=frame")

def video_stream():
    """ Generate video frames with mask detection applied. """
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not access webcam")
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = mask_detection(frame)
        _, buffer = cv.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
def live_construction(request):
    """ Render the home page. """
    return render(request, 'live.html')