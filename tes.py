import cv2
vid = cv2.VideoCapture(0)  # Open the default camera
while True:
    ret, frame = vid.read()  # Read a frame from the camera
    cv2.imshow('frame', frame)  # Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break
vid.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows