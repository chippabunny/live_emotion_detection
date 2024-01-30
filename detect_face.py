import cv2
from tensorflow.keras.models import model_from_json
import numpy as np



def facialexpressionmodel(json_file,weights_file):
    with open(json_file,'r') as file:
        loaded_model=file.read()
        model=model_from_json(loaded_model)
    model.load_weights(weights_file)
    model.compile(optimizer='adam',loss='categorical crossentropy',metrics=['accuracy'])
    return model

emotions_list=['angry','disgust','fear','happy','neutral','sad','surprise']
model=facialexpressionmodel('model_architecture.json','model_weights.h5')
faceCascade = cv2.CascadeClassifier("haarcascades_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)


def detect(file_path):
    global label
    image=cv2.imread(file_path)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray_image,1.2,3,5)

    pred=""
    try:
        for (x,y,w,h) in faces:
            fc=gray_image[y:y+h,x:x+w]
            roi=cv2.resize(fc,(48,48))
            pred=emotions_list[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print("predicted emotion is "+pred)
    except:
        pass
    

    return pred



wait=0
emotion=""
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    key=cv2.waitKey(100)
    wait+=100


    if key==ord('q'):
        break
    if wait==5000:
        filename="frame"+".jpg"
        cv2.imwrite(filename,frame)
        emotion=detect(filename)
        wait=0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
