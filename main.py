import cv2
import datetime

# Initialize the camera
cap = cv2.VideoCapture(5)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False
record_start_time = None

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Failed to read the first frame.")
    exit()

# Convert frame to grayscale and apply Gaussian blur
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    # Read next frame
    ret, frame2 = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale and apply Gaussian blur
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Calculate the absolute difference between the current frame and the background frame
    delta_frame = cv2.absdiff(gray1, gray2)

    # Threshold the delta frame
    thresh = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours on the thresholded image
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in cnts:
        if cv2.contourArea(contour) < 500:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if motion_detected:
        if not recording:
            recording = True
            record_start_time = datetime.datetime.now()
            video_filename = record_start_time.strftime("%Y%m%d_%H%M%S.avi")
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        print("Motion detected! Recording...")
    if recording:
        out.write(frame2)
        if (datetime.datetime.now() - record_start_time).seconds >= 30:
            recording = False
            out.release()
            print("Recording stopped.")

    # Show the current frame and the thresholded image
    cv2.imshow('Frame', frame2)
    cv2.imshow('Threshold', thresh)

    # Update the background frame
    gray1 = gray2

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup: close the window and release the capture
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
