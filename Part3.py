import cv2
import numpy as np
import datetime

def convolution(image, kernel):
    """
    Apply convolution to an image using a specified kernel.
    
    Args:
        image: Input image as a NumPy array.
        kernel: Convolution kernel as a NumPy array.
        
    Returns:
        Convolved image as a NumPy array.
    """
    # Get image and kernel dimensions
    img_height, img_width = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2

    # Create output image with same dimensions as input
    output = np.zeros_like(image, dtype=np.float32)

    # Pad the image with edge reflection
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Perform convolution
    for y in range(img_height):
        for x in range(img_width):
            # Extract the region of interest
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # Apply the kernel to the regionn 
            output[y, x] = np.sum(region * kernel)

    return output

# Define kernels for different filters
kernels = [
    ('Original', None),
    ('Box Filter', np.ones((3, 3)) / 9),
    ('Gaussian Filter', np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16),
    ('Sobel X', np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])),
    ('Sobel Y', np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
    ('Sharpen', np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    ('Emboss', np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])),
    ('Edge Detect', np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])),
    ('Outline', np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
]

def process_frame(frame, current_filter):
    """Helper function to process a single frame with the current filter."""
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get current filter settings
    filter_name, kernel = kernels[current_filter]
    
    # Apply filter
    if kernel is None:
        filtered_frame = gray_frame
    else:
        filtered_frame = convolution(gray_frame, kernel)
        filtered_frame = np.clip(filtered_frame, 0, 255).astype(np.uint8)
    
    # Add filter name and instructions
    cv2.putText(filtered_frame, f"Filter: {filter_name}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(filtered_frame, "Space: Next filter | Q: Quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Create original display with text
    og_display = frame.copy()
    cv2.putText(og_display, "Original", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return filter_name, filtered_frame, og_display

def main():
    """
    Main function to capture video and apply filters in real-time.
    Controls:
    - Space: Switch to next filter
    - S: Save current frame
    - Q or ESC: Quit
    """
    # Variable to store camera object
    cap = None
    
    try:
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Using camera index: {camera_index}")
                break
        else:
            print("Error: No camera was found.")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize filter index
        current_filter = 0

        # Print controls
        print("\n=== Filter controls ===")
        print("Space: Switch to next filter")
        print("S: Save current frame")
        print("Q: Quit")
        print("\nAvailable filters:")
        for i, (name, _) in enumerate(kernels):
            print(f"{i}: {name}")

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't read frame")
                break

            # Process frame
            filter_name, filtered_frame, og_display = process_frame(frame, current_filter)

            # Combine images horizontally
            combined_display = np.hstack((og_display, cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)))

            # Show result
            cv2.imshow("Webcam Filter", combined_display)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("\nExiting...")
                break
            elif key == ord(' '):  # space
                current_filter = (current_filter + 1) % len(kernels)
                print(f"Switched to: {kernels[current_filter][0]}")
            elif key == ord('s'):  # s
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"filtered_{filter_name.replace(' ', '_').lower()}_{timestamp}.png"
                cv2.imwrite(filename, filtered_frame)
                print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Clean up
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("End of filter session")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        cv2.destroyAllWindows()