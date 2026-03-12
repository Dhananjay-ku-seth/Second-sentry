import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

print("Camera opened")

# Skip first frames (camera warm-up)
for i in range(10):
    cap.read()

ret, frame = cap.read()

if ret:
    print("Frame captured successfully")
    cv2.imwrite("test.jpg", frame)
    print("Image saved as test.jpg")
else:
    print("Failed to capture frame")

cap.release()
