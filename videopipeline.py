import cv2

keys = ["A", "Am", "B", "Bm", "C", "Cm", "D", "Dm", "E", "Em", "F", "Fm", "G", "Gm"]

#it seems unlikely the middle pixel method will work well as a classifier. 
def middle_pixel(frame):
    width, height, _ = frame.shape
    return (frame[height // 2][width // 2][0] // 20) % 14

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    text = keys[middle_pixel(frame)]
    location = (50, 100)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (0, 0, 0)
    thickness = 2
    lineType = cv2.LINE_AA

    cv2.putText(frame, text, location, fontFace, fontScale, color, thickness, lineType)

    cv2.imshow('Laptop Camera Feed', frame)

    #q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()




