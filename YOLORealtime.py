import cv2

cap = cv2.VideoCapture(0)

print("Opened:", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame gagal dibaca")
        break

    cv2.imshow("Logitech C270 Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()