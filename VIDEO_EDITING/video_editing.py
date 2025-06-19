import os
import cv2
import csv
import numpy as np
import logging
import random
import shutil
from PIL import Image, ImageDraw, ImageFont
from pilmoji import Pilmoji
from moviepy import *

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Folder paths and common parameters
# Text Related Parameters
font_path = 'VIDEO_EDITING/FONTS/seguiemj_bold.ttf'
font_size = 45

# Video Related Parameters
scale = 1
video_folder_path = 'VIDEOS'
reel_folder_path = 'REELS'
reel_upload_folder_path = 'REEL_TO_UPLOAD'
model_path = 'VIDEO_EDITING/MODEL/ESPCN_x4.pb'  # Path to the super-resolution model
SuperResolution = cv2.dnn_superres.DnnSuperResImpl_create()
SuperResolution.readModel(model_path)
SuperResolution.setModel("espcn", scale)

# Image Related Parameters
num = random.randint(1,2)
overlay_image_path = f'VIDEO_EDITING/IMAGES/up_overlay_image_{num}.jpg'

# Text/csv related parameters
counter_file = 'VIDEO_EDITING/../counter.txt'
csv_file_path = 'VIDEO_EDITING/extracted_texts.csv'     
number_1_path = 'VIDEO_EDITING/number_1.txt'
number_2_path = 'VIDEO_EDITING/number_2.txt'
number_3_path = 'VIDEO_EDITING/number_3.txt'

# Reel Related Paramters      
crop_margin = 2
crop_x, crop_y, crop_w, crop_h = 0, 0, 0, 0
reel_width, reel_height = 1080, 1920

def get_reel_number():
    """Read the current counter value from the file, or initialize it."""
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as file:
            return int(file.read())
    return 0

def remove_previous_reel(reel_number):
    """
    Remove the previous day's reel from the REEL_TO_UPLOAD folder.
    
    Parameters:
        reel_number (int): The current reel number.
    """
    if reel_number > 1:
        previous_reel_path = os.path.join(reel_upload_folder_path, f'reel_{reel_number - 1}.mp4')
        if os.path.exists(previous_reel_path):
            os.remove(previous_reel_path)
            logging.info(f"Removed previous day's reel: {previous_reel_path}")
        else:
            logging.warning(f"Previous day's reel not found: {previous_reel_path}")

def get_input_video(reel_number):
    ## Loop over the video folder and get the input video path
    for filename in os.listdir(video_folder_path):
        if filename.startswith("Video") and filename.endswith(f"_{reel_number}.mp4"):
            input_video_path = os.path.join(video_folder_path, filename) 
            logging.info(f'Processing {filename} as reel_{reel_number}')

    return input_video_path

def copy_to_upload_folder(current_reel_path, reel_number):
    """
    Copy the current day's reel to the REEL_TO_UPLOAD folder.
    
    Parameters:
        current_reel_path (str): Path of the current day's processed reel.
        reel_number (int): The current reel number.
    """
    if not os.path.exists(reel_upload_folder_path):
        os.makedirs(reel_upload_folder_path)
    destination_path = os.path.join(reel_upload_folder_path, f'reel_{reel_number}.mp4')
    shutil.copy(current_reel_path, destination_path)  # Move the file to avoid duplicates
    logging.info(f"Copied reel_{reel_number} to REEL_TO_UPLOAD folder.")

def check_number_in_file(number_txt_file_path,number_to_check):
    try:
        with open(number_txt_file_path, 'r') as file:
            numbers = {line.strip() for line in file}  # Using a set for fast lookup

        if str(number_to_check) in numbers:
            print(f"Number {number_to_check} is present in the file. Performing Action 1.")
            # Action 1: Replace this with what you want to do
            return True
        else:
            print(f"Number {number_to_check} is NOT present in the file. Performing Action 2.")
            # Action 2: Replace this with what you want to do
            return False

    except FileNotFoundError:
        print("File not found. Please check the file path.")
        
def detect_video_area(frame,reel_number):
    """
    Detect the embedded video area within a frame.
    
    Parameters:
        frame (numpy.ndarray): Input video frame.
    
    Returns:
        tuple: Cropped area's (x, y, w, h), or None if detection fails.
    """
    max_x, max_y = 0, 0
    min_w, min_h = reel_width, reel_height
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(check_number_in_file(number_1_path,reel_number)):
        # Step 2: Improve contrast using histogram equalization
        gray = cv2.equalizeHist(gray)
            
        # Step 3: Multi-Scale Edge Detection (Canny + Laplacian)
        high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        canny_edges = cv2.Canny(gray, low_thresh, high_thresh)
        laplacian_edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=7)

        # Step 4: Combine Edge Maps
        edges = cv2.bitwise_or(canny_edges, laplacian_edges)

        margin = 10  # Reduced margin for more accuracy
    else:
        edges = cv2.Canny(gray, 0, 250)
        margin = 0  # Reduced margin for more accuracy

    # Step 5: Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Step 6: Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 7: Filter out small/noisy contours by setting a minimum area threshold
        min_contour_area = 0.25 * frame.shape[0] * frame.shape[1]  # 5% of the frame size
        if cv2.contourArea(largest_contour) < min_contour_area:
            return None

        # Step 8: Compute bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Step 9: Add padding/margin and ensure boundaries are within the frame
        
        max_x = max(max_x, x + margin)
        min_w = min(min_w, w - 2*margin)
        max_y = max(max_y, y + margin)
        min_h = min(min_h, h - 2*margin)
        if(check_number_in_file(number_2_path,reel_number)):
            margin = 60
            max_y = max(max_y, y + margin)
            min_h = min(min_h, h - 2*margin)
        elif(check_number_in_file(number_3_path,reel_number)):
            margin = 90
            max_y = max(max_y, y + margin)
            min_h = min(min_h, h - 2*margin)
        return max_x, max_y, min_w, min_h

    return None

# Function to enhance the sharpness of the image
def sharpen_image(frame):
    logging.debug("Applying sharpening filter.")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

# Function to adjust the contrast of the image
def adjust_contrast(frame, alpha=1.2, beta=0):
    logging.debug("Adjusting contrast and brightness.")
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Function to enhance color
def enhance_color(frame):
    logging.debug("Enhancing color by converting to HSV and adjusting saturation.")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.addWeighted(hsv[:, :, 1], 1.3, hsv[:, :, 1], 0, 0)  # Increase saturation by 30%
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def enhance_video(frame):
    return SuperResolution.upsample(frame)

def read_text_from_csv(reel_number):
    with open(csv_file_path, 'r', encoding='utf-8') as file:  # Ensure UTF-8 encoding
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['Reel Number']) == reel_number:
                return row['Text']
    return ""
    
def wrap_text(font, max_width,reel_number):
    pil_image = Image.new('RGB', (max_width, reel_height), (0, 0, 0))
    draw = ImageDraw.Draw(pil_image)
    text_to_overlay = read_text_from_csv(reel_number)
    words = text_to_overlay.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = f"{current_line} {word}".strip()
        width = draw.textbbox((0, 0), test_line, font=font)[2]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    for x in lines:
        x = x.encode('utf-8')
        print("x = ",x)
        
    return "\n".join(lines)

def make_emoji_image(reel_number):
    emoji_font = ImageFont.truetype(font_path, font_size)

    # Create image to calculate text size and handle wrapping
    image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    with Pilmoji(image) as pilmoji:
        # Break the emoji text into lines if it's too wide for the video
        wrapped_text = wrap_text(emoji_font, reel_width - 200, reel_number)
        lines = wrapped_text.split("\n")
        max_width = reel_width - 200  # Use the maximum width for alignment
        text_height = sum(pilmoji.getsize(line, emoji_font)[1] for line in lines)

    # Create a new image with the calculated size
    image = Image.new("RGBA", (max_width, text_height), (0, 0, 0, 0))

    # Define some neon colors
    neon_colors = [
        (11,255,0),  # Neon green
        (255,23,112), # Deep pink
        (0,246,255),  # Cyan
        (255,241,0),  # Yellow
        (255,0,0) # Red
    ]

    with Pilmoji(image) as pilmoji:
        y_offset = 0
        for line in lines:
            words = line.split(' ')
            line_width = sum(pilmoji.getsize(word + ' ', emoji_font)[0] for word in words)
            x_offset = (max_width - line_width) // 2  # Center align the entire line

            # Choose a random word index to color
            random_word_index = random.randint(0, len(words) - 1)
            neon_color = random.choice(neon_colors)

            for i, word in enumerate(words):
                word_width, word_height = pilmoji.getsize(word + ' ', emoji_font)
                # Apply neon color to the randomly chosen word, otherwise use white
                color = neon_color if i == random_word_index else (255, 255, 255)
                pilmoji.text((x_offset, y_offset), word + ' ', color, emoji_font, emoji_scale_factor=0.9)
                x_offset += word_width
            y_offset += word_height

    return np.array(image)
            
def clean_up_files(*files):
    """Delete intermediate video files after the final video is created."""
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted file: {file_path}")
        else:
            logging.warning(f"File not found, skipping deletion: {file_path}")
            
def process_video(input_video_path, reel_number):
    cropped_video_path = os.path.join(reel_folder_path, f'cropped_video_no_audio_{reel_number}.mp4')
    cropped_video_with_audio_path = os.path.join(reel_folder_path, f'cropped_video_{reel_number}.mp4')
    output_reel_path = os.path.join(reel_folder_path, f'reel_no_audio_{reel_number}.mp4')
    video_overlay_path = os.path.join(reel_folder_path, f'reel_video_{reel_number}.mp4')
    text_overlay_path = os.path.join(reel_folder_path, f'reel_text_{reel_number}.mp4')
    image_overlay_path = os.path.join(reel_folder_path, f'reel_image_{reel_number}.mp4')
    final_output_path = os.path.join(reel_folder_path, f'reel_{reel_number}.mp4')
    
    def add_audio_to_video(input_path, output_path):
        with VideoFileClip(input_video_path) as original_clip, VideoFileClip(input_path) as cropped_clip:
            final_clip = cropped_clip.with_audio(original_clip.audio)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", bitrate="5000k", audio_bitrate="192k")
    
    def crop_video():
        global crop_x, crop_y, crop_w, crop_h
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {input_video_path}")
            return
        video_area = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if video_area is None:
                video_area = detect_video_area(frame,reel_number)
                if video_area:
                    x, y, w, h = video_area
                    crop_x, crop_y, crop_w, crop_h = x, y, w, h
                    logging.info(f"Detected video area: {video_area}")
                    
        cap.release()
        
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {input_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        x, y, w, h = crop_x, crop_y, crop_w, crop_h
        out = cv2.VideoWriter(cropped_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*scale, h*scale))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Crop the frame using the detected area
            cropped_frame = frame[y:y + h, x:x + w]
            # Apply enhancements: sharpening, contrast, and color enhancement
            # cropped_frame = sharpen_image(cropped_frame)
            # cropped_frame = adjust_contrast(cropped_frame)
            # cropped_frame = enhance_color(cropped_frame)
            # cropped_frame = enhance_video(cropped_frame)

            # Write the enhanced, cropped frame
            out.write(cropped_frame)

        cap.release()
        if out:
            out.release()
            add_audio_to_video(cropped_video_path, cropped_video_with_audio_path)
    
    def overlay_video():

        extracted_cap = cv2.VideoCapture(cropped_video_with_audio_path)
        if not extracted_cap.isOpened():
            logging.error(f"Could not open cropped video {cropped_video_with_audio_path}")
            return

        fps = extracted_cap.get(cv2.CAP_PROP_FPS)
        output_reel = cv2.VideoWriter(output_reel_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (reel_width, reel_height))
        print("reel_height = ",reel_height)
        print("reel_width = ",reel_width)

        while extracted_cap.isOpened():
            ret, frame = extracted_cap.read()
            if not ret:
                break
            frame_height, frame_width = frame.shape[:2]
            # print("frame_height = ",frame_height)
            # print("frame_width = ",frame_width)
            aspect_ratio = frame_width / frame_height
            # print("aspect_ratio = ",aspect_ratio)
            if aspect_ratio > (reel_width / reel_height):
                new_width, new_height = reel_width, int(reel_width / aspect_ratio)
            else:
                new_width, new_height = int((reel_height) * aspect_ratio), reel_height

            temp_width = new_width
            temp_height = new_height
            new_width = new_width
            new_height = new_height
            if new_width <= 0:
                new_width = temp_width
            if new_height <= 0:
                new_height = temp_height
            
            # print("new_width = ",new_width)
            # print("new_height = ",new_height)
            
            resized_frame = cv2.resize(frame, (new_width, new_height))
            # print("resized_frame = ",resized_frame)
            black_background = np.zeros((reel_height, reel_width, 3), dtype=np.uint8)
            center_x, center_y = (reel_width - new_width) // 2, int((reel_height - new_height) // 1.2)
            
            # Round the new height to the nearest 500
            rounded_height = 500 * round(new_height / 500)

            # Adjust `center_y` based on the rounded height
            if rounded_height == 500:  # Nearest to 500 or 1000
                center_y = int(center_y // 1.5)  # Replace with your desired value
            
            if rounded_height == 1000:
                center_y = int(center_y // 1.2)  # Replace with your desired value
            
            # print("center_x = ", center_x)
            # print("center_y = ", center_y)
            
            # Create an RGBA image for rounded corners
            rounded_frame = np.zeros((new_height, new_width, 4), dtype=np.uint8)

            # Convert frame to RGBA
            resized_frame_rgba = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2BGRA)

            # Create rounded mask
            mask = np.zeros((new_height, new_width, 4), dtype=np.uint8)
            mask[:, :, 3] = 255  # Set alpha to fully opaque

            radius = min(new_width, new_height) // 17  # Adjust roundness
            pil_mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(pil_mask)
            draw.rounded_rectangle((0, 0, new_width, new_height), radius=radius, fill=(255, 255, 255, 255))
            mask = np.array(pil_mask)

            # Apply rounded corners mask
            rounded_frame = cv2.bitwise_and(resized_frame_rgba, mask)

            # Convert back to BGR (discard alpha since black bg is used)
            rounded_frame_bgr = cv2.cvtColor(rounded_frame, cv2.COLOR_BGRA2BGR)

            # Place the rounded video on the black background
            black_background[center_y:center_y + new_height, center_x:center_x + new_width] = rounded_frame_bgr
            
            output_reel.write(black_background)

        extracted_cap.release()
        output_reel.release()
        add_audio_to_video(output_reel_path, video_overlay_path)
        return center_x,center_y

    def overlay_text(center_x,center_y):
        """
        Overlays text onto the final video.

        """

        # Load video
        video = VideoFileClip(video_overlay_path)

        # creates the text image to insert
        text_overlay_image = make_emoji_image(reel_number)

        # Resizing
        text_overlay_aspect_ratio = text_overlay_image.shape[1] / text_overlay_image.shape[0]
        text_overlay_width = text_overlay_image.shape[1]
        text_overlay_height = int(text_overlay_width / text_overlay_aspect_ratio)
        
        text_overlay_image_resized = Image.fromarray(text_overlay_image).resize((int(text_overlay_width),int(text_overlay_height)))
        
        # Convert the resized image to a numpy array again for ImageClip
        text_overlay_clip = ImageClip(np.array(text_overlay_image_resized), duration=video.duration)

        # Repositioning
        text_y = center_y - text_overlay_height - 40
        text_overlay_clip = text_overlay_clip.with_position((center_x,text_y)) 

        # Merge text image with video
        final_clip = CompositeVideoClip([video, text_overlay_clip])

        # Save output    
        final_clip.write_videofile(text_overlay_path, codec="libx264", audio_codec="aac", fps=video.fps)
        
        # Close the Clip
        final_clip.close()
        return text_y
        
    def overlay_image(text_y):
        """
        Overlays an image onto the final video.

        """
        # Load the final video
        video = VideoFileClip(text_overlay_path)

        # Load the overlay image using Pillow
        overlay_image = Image.open(overlay_image_path)

        # Convert Pillow image to a numpy array for MoviePy
        overlay_image_np = np.array(overlay_image)
        
        # Resize overlay image if needed
        overlay_image_aspect_ratio = overlay_image_np.shape[1] / overlay_image_np.shape[0]
        overlay_image_width = overlay_image_np.shape[1] - 400
        overlay_image_height = int(overlay_image_width / overlay_image_aspect_ratio)
        overlay_image = overlay_image.resize((int(overlay_image_width),int(overlay_image_height)))  # Adjust size as needed

        # Convert Pillow image to a numpy array for MoviePy
        overlay_image_np = np.array(overlay_image)

        # Create an ImageClip from the numpy array
        overlay_image_clip = ImageClip(overlay_image_np, duration=video.duration)

        # Set the overlay image position (adjust as needed)
        image_y = text_y - overlay_image_height - 5
        overlay_image_clip = overlay_image_clip.with_position(("center", image_y))

        # Merge overlay image with video
        final_clip = CompositeVideoClip([video, overlay_image_clip])

        # Save the output video
        final_clip.write_videofile(image_overlay_path, codec="libx264", audio_codec="aac", fps=video.fps)

        # Close the Clip
        final_clip.close()
    
    # Step 1 - Crop the Video Out
    crop_video()
    # Step 2 - Overlay the cropped Video on a Black Background
    # center_x, center_y = overlay_video()
    # # Step 3 - Overlay Text
    # text_y = overlay_text(center_x,center_y)
    # # Step 4 - Overlay Image
    # overlay_image(text_y)
    # # Step 6 - Clean up some files
    # shutil.copy(image_overlay_path,final_output_path)
    # clean_up_files(cropped_video_path, cropped_video_with_audio_path, output_reel_path, video_overlay_path, text_overlay_path, image_overlay_path)
    clean_up_files(cropped_video_path)
def main():
    # Getting the reel number from counter file.
    # reel_number = get_reel_number()
    # print(f"Reel Number : {reel_number}")
    
    # Remove the previous day's reel before starting today's processing
    # remove_previous_reel(reel_number)
    
    try:
        for reel_number in range(151, 331):
            # Get the input video from video folder.
            input_video_path = get_input_video(reel_number)
            
            # Process the input video
            process_video(input_video_path, reel_number)
    except Exception as e:
        logging.error(f"An error occurred during batch processing: {e}")
            
    # # Copy the final processed reel to REEL_TO_UPLOAD
    # final_reel_path = os.path.join(reel_folder_path, f'reel_{reel_number}.mp4')
    # copy_to_upload_folder(final_reel_path, reel_number)
            
if __name__ == "__main__":
    main()
