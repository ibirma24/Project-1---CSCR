import cv2
import numpy as np
import datetime

def apply_convolution(image, filter):
    """
    Apply convolution to an image using a specified filter.
    
    Args:
        image: Input image as a NumPy array.
        filter: Convolution filter as a NumPy array (must be square and odd-sized).
        
    Returns:
        Convolved image as a NumPy array.
    """
    # Get image and filter dimensions
    img_height, img_width = image.shape
    filter_size = filter.shape[0]
    pad_size = filter_size // 2

    # Create output image with same dimensions as input
    output = np.zeros_like(image, dtype=np.float32)

    # Pad the image with edge reflection
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Perform convolution
    for y in range(img_height):
        for x in range(img_width):
            # Extract the region of interest
            region = padded_image[y:y + filter_size, x:x + filter_size]
            # Apply the filter to the region
            output[y, x] = np.sum(region * filter)

    return output

# Define kernels for different filters
kernels = [
    ('Original', None),
    ('Box Filter', np.ones((3, 3)) / 9),
    ('Gaussian Filter', np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16),
    ('Sobel X', np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])),
    ('Sharpen', np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
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
        filtered_frame = apply_convolution(gray_frame, kernel)
        filtered_frame = np.clip(filtered_frame, 0, 255).astype(np.uint8)
    
    # Create original display with text
    og_display = frame.copy()
    cv2.putText(og_display, "Original Feed", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(og_display, "Space: Next filter | Q: Quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add text to filtered frame
    cv2.putText(filtered_frame, f"Filter: {filter_name}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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

        # Set window positions for side-by-side display
        cv2.namedWindow('Original Feed')
        cv2.namedWindow('Filtered Output')
        cv2.moveWindow('Original Feed', 50, 100)
        cv2.moveWindow('Filtered Output', 700, 100)  # Position to the right
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

            # Show original and filtered images in separate windows
            cv2.imshow('Original Feed', og_display)
            cv2.imshow('Filtered Output', filtered_frame)

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