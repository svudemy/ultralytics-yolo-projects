import cv2
from ultralytics import YOLO
import numpy as np
import time
import argparse
import os
import math

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="yolo11l-pose.pt", help="YOLO11 Pose model path")
parser.add_argument("--video", type=str, default="inference/videos/bicep-curl.mp4", help="Path to input video or webcam index (0)")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold")
parser.add_argument("--save", action="store_true", help="Save the result")  # Option to save the output video
args = parser.parse_args()

# Function to display FPS on the frame
def show_fps(frame, fps):
    # Get the dimensions of the frame (height and width)
    height, width = frame.shape[:2]

    # Define the top-left corner position (x, y) of the FPS display box
    # and calculate its width and height as a proportion of the frame size
    x, y = 10, 10
    w = int(width * 0.25)  # Box width is 25% of the frame width
    h = int(height * 0.08)  # Box height is 8% of the frame height

    # Format the FPS text to be displayed
    text = "FPS: " + str(fps)

    # Set the font type and calculate the font size and thickness
    font = cv2.FONT_HERSHEY_PLAIN  # Use a simple font style
    font_scale = w * 0.01  # Font scale is proportional to the box width
    font_thickness = math.ceil(w * 0.01)  # Font thickness is proportional to the box width

    # Calculate the size of the text for proper centering
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]  # Get text dimensions (width, height)

    # Calculate the position to center the text within the box
    text_x = x + (w - text_size[0]) // 2  # Center horizontally
    text_y = y + (h + text_size[1]) // 2  # Center vertically

    # Draw a filled rectangle as the background for the FPS text
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Black rectangle with no border

    # Add the FPS text on top of the rectangle
    cv2.putText(frame, "FPS: " + str(fps), (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 255, 0), font_thickness)

# Function to display Counter on the frame
def show_counter(frame, title, position, count):    
    height, width = frame.shape[:2]

    # Define the size of the box    
    box_width = int(width * 0.15)
    box_height = int(height * 0.1)

    # Coordinates for the center top of the frame
    center_x = width // 2
    top_y = 10  # The y-coordinate position for the top of the box

    # Box color (default to green)
    box_color = (0, 255, 0)

    # Determine box position (left or right) and set color
    if position == 'left':        
        start_x = center_x - box_width - 10  # Offset 10 pixels between boxes
    else:  # position == 'right'
        start_x = center_x +    10  # Offset 10 pixels for the right box

    # Set starting and ending coordinates for the box
    start_y = top_y
    end_x = start_x + box_width
    end_y = start_y + box_height

    # Draw the box on the frame using cv2.rectangle
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, -1)  # Filled box

    # Create the text to display, based on the box type (IN or OUT)
    text = f"{title}: {count}"
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = box_width * 0.01
    font_thickness = math.ceil(box_width * 0.02)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Coordinates to center the text inside the box
    text_x = start_x + (box_width - text_size[0]) // 2
    text_y = start_y + (box_height + text_size[1]) // 2

    # Draw the text inside the box using cv2.putText
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

# Function to draw keypoints and their connecting line, along with the calculated angle
def draw_keypoints_angle(frame, keypoint_ids, keypoints, angle):
    # Initialize an empty list to store coordinates of the specified keypoints
    keypoints_coord = []

    # Loop through the list of keypoint IDs to extract their coordinates
    for keypoint_id in keypoint_ids:
        # Append the keypoint's coordinates to the list
        keypoints_coord.append(keypoints[0][keypoint_id])

        # Extract the x and y coordinates of the keypoint
        x, y = keypoints[0][keypoint_id]

        # Draw a circle at the keypoint position on the frame
        object_point = (int(x), int(y))  # Convert coordinates to integers
        cv2.circle(frame, object_point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

    # Convert the list of keypoint coordinates to a numpy array for drawing lines
    points = np.array(keypoints_coord, dtype=np.int32)

    # Draw a polyline connecting the keypoints (not closed)
    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)  # Green line

    # Extract the coordinates of the second keypoint (used for placing the angle text)
    x, y = keypoints_coord[1]
    elbow_point = (int(x), int(y))  # Convert coordinates to integers

    # Format the angle value to two decimal places
    angle_float = float("{:.2f}".format(angle))

    # Display the angle value near the second keypoint
    cv2.putText(frame, str(angle_float), elbow_point, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) 

# Function to calculate the angle between three keypoints: shoulder, elbow, and wrist
def calculate_angle(keypoint_ids, keypoints):
    # Initialize an empty list to store the coordinates of the specified keypoints
    keypoints_coord = []

    # Loop through the list of keypoint IDs and extract the corresponding coordinates
    for keypoint_id in keypoint_ids:
        # Append the coordinates of each keypoint to the list
        keypoints_coord.append(keypoints[0][keypoint_id])

    # Assign the coordinates to the respective body parts: shoulder, elbow, wrist
    shoulder, elbow, wrist = keypoints_coord

    # Calculate the angle between the shoulder, elbow, and wrist using the atan2 function
    # atan2 returns the angle between the positive x-axis and the line to the point (x, y)
    radians = math.atan2(wrist[1]-elbow[1], wrist[0]-elbow[0]) - math.atan2(shoulder[1]-elbow[1], shoulder[0]-elbow[0])

    # Convert the angle from radians to degrees
    angle = math.degrees(radians)

    # Take the absolute value of the angle to ensure it's positive
    angle = abs(angle)

    # If the angle is greater than 180 degrees, adjust it by subtracting from 360
    if angle > 180.0:
        angle = 360 - angle    

    # Return the calculated angle
    return angle

# Function to check if the keypoints are visible (i.e., not equal to 0)
def is_keypoints_visible(keypoints, keypoint_ids):
    # Loop through the provided keypoint IDs
    for keypoint_id in keypoint_ids:        
        # Extract the x, y coordinates of the keypoint
        x, y = keypoints[0][keypoint_id]
        # If either x or y is 0, return False indicating the keypoint is not visible
        if(int(x) == 0 or int(y) == 0):
            return False    
    # If all keypoints are visible, return True
    return True

# Function to count the number of repetitions based on the angle and previous condition
def count_repetitions(angle, prev_condition, count):
    # If the angle is less than 50, it indicates a "down" position (e.g., squat down)
    if(angle < 50):
        # If the previous condition was "down", count a repetition and set the condition to "up"
        if(prev_condition == "down"):
            count += 1        
        prev_condition = "up"  # After the "down" condition, the next state is "up"

    # If the angle is greater than 120, it indicates an "up" position (e.g., squat up)
    if(angle > 120):
        prev_condition = "down"  # After the "up" position, the next state is "down"

    # Return the updated condition and count
    return prev_condition, count    

# Function to adjust the aspect ratio of the frame to fit the screen size
def adjust_aspect_ratio(frame, screen_width, screen_height):
    # Get the current height, width, and channels of the frame
    h, w, _ = frame.shape
    aspect_ratio = w / h  # Calculate the aspect ratio of the frame

    # If the screen's aspect ratio is greater than the frame's aspect ratio
    if screen_width / screen_height > aspect_ratio:
        # Fit the frame height to the screen height, and adjust width proportionally
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)
    else:
        # Fit the frame width to the screen width, and adjust height proportionally
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)

    # Resize the frame to fit the screen while maintaining the aspect ratio
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Calculate the padding required to center the frame in the screen
    top = (screen_height - new_height) // 2
    bottom = screen_height - new_height - top
    left = (screen_width - new_width) // 2
    right = screen_width - new_width - left

    # Add black padding around the resized frame to center it on the screen
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Return the padded frame with centered content
    return padded_frame

if __name__ == '__main__':
    # Set up video capture
    video_input = args.video
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)  # Open webcam if video_input is a digit
    else:
        cap = cv2.VideoCapture(video_input)  # Open video file
    
    # Save Video
    output_folder = "result"  # Directory to save the output video
    if(not os.path.isdir(output_folder)):  # Create the directory if it doesn't exist
        os.mkdir(output_folder)

    if args.save:  # If the save option is selected
        # Extract the filename from the input video and remove the extension
        filename = os.path.splitext(os.path.basename(args.video))[0]

        # Define the path for the output video
        output_video_path = f"{output_folder}/{filename}.mp4"  

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the input video

        # Create video writer objects to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # Initialize the VideoWriter

    conf_thres = args.conf  # Confidence threshold for detection
    model = YOLO(args.model)  # Load YOLOv11 model        

    # Initialize the count for the right hand repetitions
    right_count = 0
    # Define the keypoint IDs for the right hand (e.g., shoulder, elbow, wrist)
    right_hand = [6, 8, 10]
    # Set the initial condition for the right hand to "down" (representing the downward position of the hand)
    right_prev_condition = "down"

    # Initialize the count for the left hand repetitions
    left_count = 0
    # Define the keypoint IDs for the left hand (e.g., shoulder, elbow, wrist)
    left_hand = [5, 7, 9]
    # Set the initial condition for the left hand to "down" (representing the downward position of the hand)
    left_prev_condition = "down"

    start_time = 0  # Initialize start time for FPS calculation

    # Start a loop that continues as long as the video capture is open
    while cap.isOpened():
        # Read a frame from the video capture object
        success, frame = cap.read()  
        # Initialize a variable to store the annotated frame (initially the same as the original frame)
        annotated_frame = frame

        # If the frame was successfully read
        if success:        
            # Perform Pose Estimation using YOLO11 (or a similar model) with a confidence threshold
            results = model(frame, conf=conf_thres, verbose=False)  
            # Extract keypoints (body part positions) from the pose estimation result
            keypoints = results[0].keypoints.xy.cpu()

            # Check if the right hand keypoints are visible (not 0,0)
            if is_keypoints_visible(keypoints, right_hand):                             
                # Calculate the angle for the right hand
                angle = calculate_angle(right_hand, keypoints)
                # Draw the keypoints and the angle on the annotated frame
                draw_keypoints_angle(annotated_frame, right_hand, keypoints, angle)

                # Update the previous condition and count repetitions for the right hand
                right_prev_condition, right_count = count_repetitions(angle, right_prev_condition, right_count)
            
            # Check if the left hand keypoints are visible (not 0,0)
            if is_keypoints_visible(keypoints, left_hand):
                # Calculate the angle for the left hand
                angle = calculate_angle(left_hand, keypoints)
                # Draw the keypoints and the angle on the annotated frame
                draw_keypoints_angle(annotated_frame, left_hand, keypoints, angle) 

                # Update the previous condition and count repetitions for the left hand
                left_prev_condition, left_count = count_repetitions(angle, left_prev_condition, left_count)            

            # Show the updated counts for both the right and left hands on the frame
            show_counter(annotated_frame, "Right", "left", right_count)            
            show_counter(annotated_frame, "Left", "right", left_count)

            # Calculate FPS (Frames Per Second) based on the time difference between frames
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            # Update the start time for the next frame calculation
            start_time = end_time

            # Format the FPS to 2 decimal places
            fps = float("{:.2f}".format(fps))
            # Show the FPS on the frame
            show_fps(annotated_frame, fps)
            
            # Display the annotated frame with all keypoints, angles, and counters
            cv2.namedWindow("YOLO11 Bicep Curl Counter", cv2.WND_PROP_FULLSCREEN)  # Create a named window for fullscreen display
            cv2.setWindowProperty("YOLO11 Bicep Curl Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set window to fullscreen

            # Adjust the aspect ratio of the annotated frame to fit the screen size (1920x1080)
            fullscreen_frame = adjust_aspect_ratio(annotated_frame, 1920, 1080)

            # Show the frame in the fullscreen window
            cv2.imshow("YOLO11 Bicep Curl Counter", fullscreen_frame)          

            # If the save option is enabled, write the annotated frame to the output video
            if args.save:  
                writer.write(annotated_frame)  # Save the annotated frame to the output video file  

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # If the video reaches its end or there's an issue reading a frame, break the loop
            break       

    # After the loop ends, print the location where the tracking results will be saved (if save option is enabled)
    if args.save:        
        print("The result will be saved in: " + output_video_path)  

    # Release the video capture object and close any open display windows
    cap.release()
    cv2.destroyAllWindows()

