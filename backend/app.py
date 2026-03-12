import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

from backend.recognition import recognize_faces

app = FastAPI()

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        results = recognize_faces(frame)

        for (x,y,w,h,name) in results:

            color = (0,255,0)

            if name == "UNKNOWN":
                color = (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(frame,name,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.get("/")
def home():

    return HTMLResponse("""
    <html>
    <body>
    <h1>Second Sentry AI Camera</h1>
    <img src="/video">
    </body>
    </html>
    """)

@app.get("/video")
def video():

    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")
