import cv2

cap = cv2.VideoCapture(0)

while True:
    ok , frame = cap.read()

    if not ok:
        print("couldn't read from camera")
        break
    frame = cv2.flip(frame, 1)

    cv2.imshow("face-unlock", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()