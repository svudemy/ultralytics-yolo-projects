import cv2
from ultralytics import YOLO
import time
import math
import argparse
import os
import telepot
import threading
from config import *

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--poseModel", type=str, default="models/yolo26n-pose.pt", help="YOLO26 Pose model")  # Path to the YOLO Pose model
parser.add_argument("--detectionModel", type=str, default="models/yolo26s-knife.pt", help="YOLO26 Weapon Detection Model")  # Path to the YOLO Detection model
parser.add_argument("--video", type=str, default="inference/suspicious.mp4", help="Path to input video or webcam index (0)")  # Input video file or webcam
parser.add_argument("--conf", type=float, default=0.45, help="Confidence Threshold")  # Confidence threshold
args = parser.parse_args()  # Parse command-line arguments

last_sent_time = LAST_SENT_TIME  # Store the timestamp of the last alert message sent
cooldown = COOLDOWN  # Time interval (in seconds) before sending another alert to prevent spam

# Function to display FPS (Frames Per Second) on the frame
def show_fps(frame, fps):
    x, y, w, h = 10, 10, 350, 50  # Define the position and size of the FPS display area
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Draw a black rectangle for background
    cv2.putText(frame, "FPS: " + str(fps), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 255, 0), 3)  # Add FPS text to the frame

def show_information(frame, title):    
    # Define the size of the box
    box_width = 800
    box_height = 50

    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Coordinates for the center of the frame
    center_x = frame_width // 2 
    center_y = 10  # The y-coordinate position for the top of the box    

    # Set starting and ending coordinates for the box (centered)
    start_x = center_x - box_width // 2
    start_y = center_y
    end_x = start_x + box_width
    end_y = start_y + box_height

    # Draw the box on the frame using cv2.rectangle
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), -1)  # Filled box

    # Create the text to display, based on the box type (IN or OUT)
    text = f"{title}"    
    
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3.5
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Coordinates to center the text inside the box
    text_x = start_x + (box_width - text_size[0]) // 2
    text_y = start_y + (box_height + text_size[1]) // 2

    # Draw the text inside the box using cv2.putText
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    # Check if any of the points is (0,0); if so, return 0 as the angle
    if any(point == (0, 0) for point in [a, b, c]):
        return 0        
    
    # Compute vectors AB and BC
    ab = [a[0] - b[0], a[1] - b[1]]  # Vector from B to A
    bc = [c[0] - b[0], c[1] - b[1]]  # Vector from B to C
    
    # Ensure neither vector is a zero vector to avoid division errors
    if all(value != 0 for value in ab) and all(value != 0 for value in bc):
        # Compute the dot product of vectors AB and BC
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        
        # Compute the magnitudes (lengths) of vectors AB and BC
        mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        
        # Compute the cosine of the angle using the dot product formula
        cos_theta = dot_product / (mag_ab * mag_bc)

        # Ensure the cosine value is within the valid range [-1,1] to prevent errors in acos
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        # Compute the angle in radians and convert it to degrees
        angle = math.acos(cos_theta)
        
        return math.degrees(angle)
    else:
        return 0  # Return 0 if any vector is invalid 

# Function to extract a keypoint based on its index and confidence
def get_keypoint(keypoint, conf, idx, threshold=0.5):
    keypoint = keypoint[idx]  # Extract the keypoint at the specified index
    if conf[idx] > threshold:  # Check if the confidence of the keypoint is above the threshold
        return keypoint            

    return (0, 0)  

# Function to draw a bounding box on the annotated frame
def draw_bbox(box, annotated_frame):
    # Extract width and height of the bounding box
    width = box[2]
    height = box[3]

    # Convert center-based coordinates to top-left corner coordinates
    x = box[0] - int(width / 2)
    y = box[1] - int(height / 2)                

    # Draw a red bounding box (BGR: (0, 0, 255)) with a thickness of 3 pixels
    cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 0, 255), 3)

# Function to send an alert message and an image using a bot
def send_alert(bot, receiver_id, frame, message):
    global last_sent_time  # Use global variable to track the last sent time
    
    # Check if the cooldown period has passed since the last alert
    if time.time() - last_sent_time < cooldown:
        return  # Exit the function if cooldown is still active

    # Update the last sent time
    last_sent_time = time.time()  

    # Save the frame as an image file
    filename = "savedImage.jpg"
    cv2.imwrite(filename, frame)
    
    try:
        # Send the alert message to the specified receiver
        bot.sendMessage(receiver_id, message)                     

        # Send the captured image to the receiver
        bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
    finally:
        # Ensure the temporary image file is deleted after sending
        os.remove(filename) 

if __name__ == '__main__':
    # Set up video capture
    video_input = args.video  # Get the video input path from arguments
    if video_input.isdigit():  # Check if the input is a digit (indicating webcam)
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)  # Open webcam
    else:
        cap = cv2.VideoCapture(video_input)  # Open video file  
    
    # Initialize the Telegram bot using the provided token
    bot = telepot.Bot(TELEGRAM_TOKEN)

    conf_thres = args.conf  # Set confidence threshold for detection and pose

    model = YOLO(args.poseModel)  # Load the YOLO26 Pose model    
    detection_model = YOLO(args.detectionModel)  # Load the YOLO26 Detection model            

    start_time = time.time() # Record the start time of the program

    while cap.isOpened():  # Main loop to process video frames
        success, frame = cap.read()  # Read a frame from the video
        annotated_frame = frame  # Copy the frame for the result

        # If frame reading fails (e.g., end of video), exit the loop
        if not success:
            break

        # Perform object tracking using YOLO                  
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_thres, verbose=False)                      

        # Extract bounding boxes as integer values and convert to a list
        boxes = results[0].boxes.xywh.int().cpu().tolist()  
        # Extract keypoints (e.g., body landmarks) and convert to a list
        keypoints = results[0].keypoints.xy.cpu().tolist()  
        # Extract confidence scores for the keypoints and convert to a list
        confs = results[0].keypoints.conf.cpu().tolist()

        # Check if tracking IDs exist in the results
        if results[0].boxes.id is not None:
            # Extract tracking IDs for detected objects
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Plot the tracking results on the annotated frame (without bounding boxes)
            annotated_frame = results[0].plot(boxes=False)           
            
            # Initialize threat detection flags
            weapon_danger = False  # Indicates if a weapon is detected
            threat = False  # Indicates if a general threat is detected
            message = ""  # Message to be displayed if a threat is found

            # Iterate through detected objects, extracting their bounding boxes, keypoints, and tracking IDs
            for box, keypoint, conf, track_id in zip(boxes, keypoints, confs, track_ids):                                                                                            
                # Extract keypoints for important body parts                
                # Shoulders (Left & Right)
                left_shoulder = get_keypoint(keypoint, conf, 5)
                right_shoulder = get_keypoint(keypoint, conf, 6)

                # Wrists (Left & Right)
                left_wrist = get_keypoint(keypoint, conf, 9)
                right_wrist = get_keypoint(keypoint, conf, 10)

                # Hips (Left & Right)
                left_hip = get_keypoint(keypoint, conf, 11)
                right_hip = get_keypoint(keypoint, conf, 12)            

                # Calculate the upper body angles using three keypoints: wrist, shoulder, and hip
                upper_body_angle_left = calculate_angle(left_wrist, left_shoulder, left_hip)        
                upper_body_angle_right = calculate_angle(right_wrist, right_shoulder, right_hip)                                                                                             
                # Detect potentially dangerous behavior based on upper body angles
                upper_body_left = upper_body_angle_left > 80  # Check if the left upper body is raised
                upper_body_right = upper_body_angle_right > 80  # Check if the right upper body is raised

                # If either arm is raised above the threshold, perform weapon detection
                if(upper_body_left or upper_body_right):
                    # Perform weapon detection using YOLO
                    detection_result = detection_model(frame, conf=conf_thres, verbose=False)  # Detect objects classified as weapons
                    detection_boxes = detection_result[0].boxes.xywh.int().cpu().tolist()  # Get detected bounding boxes
                    class_boxes = detection_result[0].boxes.cls.int().cpu().tolist()  # Get class IDs of detected objects                

                    # Iterate through detected objects
                    for detection_box, class_id in zip(detection_boxes, class_boxes):                        
                        x1, y1, w, h = detection_box # Extract bounding box coordinates

                        # Extract wrist positions
                        xrw, yrw = right_wrist
                        xlw, ylw = left_wrist

                        # Calculate Euclidean distance between wrists and detected weapon
                        right_distance = math.sqrt((xrw - x1) ** 2 + (yrw - y1) ** 2)                            
                        left_distance = math.sqrt((xlw - x1) ** 2 + (ylw - y1) ** 2)                            

                        # If a detected weapon is close to either wrist, raise an alert
                        if right_distance < WEAPON_DISTANCE or left_distance < WEAPON_DISTANCE:
                            draw_bbox(detection_box, annotated_frame)  # Draw bounding box around the detected weapon
                            show_information(annotated_frame, "Weapon Threat Detected")  # Display a warning message

                            threat = True  # Mark the situation as a threat
                            message = "Weapon Threat Detected"  # Set the alert message                              
                
                # If a threat is detected, proceed with alert mechanisms
                if(threat):
                    # Draw a red bounding box around the detected threat (e.g., person carrying a weapon)
                    draw_bbox(box, annotated_frame)                       

                    # Create a separate thread to send an alert asynchronously
                    # This prevents the main program from being delayed while sending the alert
                    alert_thread = threading.Thread(target=send_alert, args=(bot, RECEIVER_ID, frame, message))
                    
                    # Start the alert thread to send the notification
                    alert_thread.start()
        
        # Get the current time to measure the time taken for processing the frame
        end_time = time.time()  

        # Calculate frames per second (FPS) using the time difference between frames
        fps = 1 / (end_time - start_time)  

        # Update start_time for the next frame to maintain accurate FPS calculation
        start_time = end_time  

        # Format FPS value to two decimal places for better readability
        fps = float("{:.2f}".format(fps))  

        # Display the FPS on the annotated frame
        show_fps(annotated_frame, fps)

        # Resize the frame to 1280x720 resolution for better visualization
        resized_frame = cv2.resize(annotated_frame, (1280, 720))  

        # Display the annotated frame in a window named "YOLO26 Suspicious Movement Detection"
        cv2.imshow("YOLO26 Suspicious Movement Detection", resized_frame)      

        # Check if the 'q' key is pressed; if so, exit the loop to stop processing
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Exit loop and terminate the program

    # Release the video capture object and close the display window
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows