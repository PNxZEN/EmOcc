"""
Webcam capture script - Press SPACE to capture and save images.
Saves images to examples folder as image0.png, image1.png, etc.
"""

import cv2
import os
import time


def get_next_image_number(save_dir):
    """Find the next available image number."""
    i = 0
    while os.path.exists(os.path.join(save_dir, f"image{i}.png")):
        i += 1
    return i


def main():
    # Configuration
    save_dir = "examples"
    camera_index = 0
    frame_width = 640
    frame_height = 480
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print("=" * 70)
    print("WEBCAM CAPTURE")
    print("=" * 70)
    print("\nControls:")
    print("  SPACE - Capture and save image")
    print("  Q     - Quit")
    print(f"\nSaving to: {os.path.abspath(save_dir)}")
    print("\n" + "=" * 70 + "\n")
    
    # Get starting image number
    image_counter = get_next_image_number(save_dir)
    captured_count = 0
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    last_capture_time = 0
    show_capture_message = False
    capture_message_time = 0
    last_saved_path = ""
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_time)
                fps_time = time.time()
                fps_counter = 0
            
            # Create display frame
            display_frame = cv2.flip(frame.copy(), 1)
            
            # Add semi-transparent overlay at top
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(display_frame, "SPACE: Capture | Q: Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display capture count
            cv2.putText(display_frame, f"Captured: {captured_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show capture confirmation message for 2 seconds
            if show_capture_message and (time.time() - capture_message_time < 2.0):
                # Add semi-transparent overlay at bottom
                overlay_bottom = display_frame.copy()
                cv2.rectangle(overlay_bottom, (0, display_frame.shape[0] - 60), 
                             (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), -1)
                cv2.addWeighted(overlay_bottom, 0.6, display_frame, 0.4, 0, display_frame)
                
                cv2.putText(display_frame, f"SAVED: {os.path.basename(last_saved_path)}", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif show_capture_message:
                show_capture_message = False
            
            # Show frame
            cv2.imshow('Webcam Capture - Press SPACE to capture, Q to quit', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Quit
                break
            elif key == ord(' '):
                # Capture image (with debouncing - minimum 0.5s between captures)
                current_time = time.time()
                if current_time - last_capture_time > 0.5:
                    # Save the original frame (not the one with overlay)
                    save_path = os.path.join(save_dir, f"image{image_counter}.png")
                    cv2.imwrite(save_path, frame)
                    
                    print(f"[{captured_count + 1}] Captured: {save_path}")
                    
                    last_saved_path = save_path
                    image_counter += 1
                    captured_count += 1
                    last_capture_time = current_time
                    show_capture_message = True
                    capture_message_time = current_time
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print(f"Session ended")
        print(f"Total images captured: {captured_count}")
        if captured_count > 0:
            print(f"Images saved in: {os.path.abspath(save_dir)}")
        print("=" * 70)


if __name__ == '__main__':
    main()
