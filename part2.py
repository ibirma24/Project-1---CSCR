import rawpy
import numpy as np
import matplotlib.pyplot as plt
import math

# Force matplotlib to show plots
plt.ion()  # Enable interactive mode

def calculate_fov(focal_length_mm, sensor_width_mm=36.0):
    """
    Calculate the horizontal field of view (FOV) in degrees.
    FOV = 2 * arctan(sensor_width / (2 * focal_length))
    
    Args:
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        
    Returns:
        Field of view in degrees
    """
    frad = 2 * math.atan(sensor_width_mm / (2 * focal_length_mm))
    fdeg = math.degrees(frad)
    return fdeg

def analyze_noise(raw_data, region=None):
    """
    Calculate noise statistics in a uniform region of the image.
    
    Args:
        raw_data: Raw image data as numpy array
        region: Tuple of (x, y, width, height) for analysis region. If None, use center region.
        
    Returns:
        Dictionary containing mean (μ), standard deviation (σ), and SNR for each color channel
    """
    if region is None:
        # Use a smaller default region (50x50 pixels) in center for more uniform area
        h, w = raw_data.shape[:2]
        x = w//2 - 25
        y = h//2 - 25
        width = 50
        height = 50
        region = (x, y, width, height)
    
    x, y, w, h = region
    roi = raw_data[y:y+h, x:x+w]
    
    stats = {}
    # Analyze each color channel separately
    for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
        if len(roi.shape) > 2:
            channel_data = roi[:,:,i]
        else:
            channel_data = roi
            
        mean = np.mean(channel_data)
        std_dev = np.std(channel_data)
        snr = mean / std_dev if std_dev != 0 else float('inf')
        
        stats[channel_name] = {
            'mean': mean,
            'std_dev': std_dev,
            'snr': snr
        }
    
    return stats

def analyze_camera_comp(main_camera_file, telephoto_file, main_focal_length, tele_focal_length, 
                       main_sensor_width, tele_sensor_width, analysis_region=None):
    """
    Compare main and telephoto camera specifications and analyze images.
    Calculates Field of View (FOV) and performs noise analysis on uniform regions.
    
    The analysis includes:
    1. FOV calculation using FOV = 2 * arctan(w/(2f)) where:
       - w is the sensor width in mm
       - f is the focal length in mm
    2. Noise analysis in uniform regions measuring:
       - Mean (μ) for light intensity
       - Standard deviation (σ) for noise level
       - Signal-to-Noise Ratio (SNR) = μ/σ
    
    Args:
        main_camera_file: Path to main camera DNG file
        telephoto_file: Path to telephoto camera DNG file
        main_focal_length: Main camera focal length in mm
        tele_focal_length: Telephoto camera focal length in mm
        main_sensor_width: Main camera sensor width in mm
        tele_sensor_width: Telephoto camera sensor width in mm
        analysis_region: Optional region for noise analysis (x, y, width, height)
    """
    # Calculate FOV for both cameras
    main_fov = calculate_fov(main_focal_length, main_sensor_width)
    tele_fov = calculate_fov(tele_focal_length, tele_sensor_width)
    
    print("\nCamera Specifications and Analysis")
    print("=================================")
    
    print(f"\nMain Camera (RCAM):")
    print(f"Focal Length: {main_focal_length} mm")
    print(f"Sensor Width: {main_sensor_width} mm")
    print(f"Field of View: {main_fov:.2f}°")
    
    print(f"\nTelephoto Camera (TCAM):")
    print(f"Focal Length: {tele_focal_length} mm")
    print(f"Sensor Width: {tele_sensor_width} mm")
    print(f"Field of View: {tele_fov:.2f}°")
    
    print(f"\nFOV Ratio (Main/Tele): {main_fov/tele_fov:.2f}")
    
    # Load and analyze images
    try:
        with rawpy.imread(main_camera_file) as raw_main:
            main_image = raw_main.postprocess()
            main_stats = analyze_noise(main_image, analysis_region)
            
        with rawpy.imread(telephoto_file) as raw_tele:
            tele_image = raw_tele.postprocess()
            tele_stats = analyze_noise(tele_image, analysis_region)
            
        print("\nNoise Analysis Results")
        print("===================")
        
        print("\nMain Camera (RCAM):")
        print("-----------------")
        for channel, stats in main_stats.items():
            print(f"{channel} Channel:")
            print(f"  Mean (μ): {stats['mean']:.2f}")
            print(f"  Standard Deviation (σ): {stats['std_dev']:.2f}")
            print(f"  Signal-to-Noise Ratio: {stats['snr']:.2f}")
        
        print("\nTelephoto Camera (TCAM):")
        print("----------------------")
        for channel, stats in tele_stats.items():
            print(f"{channel} Channel:")
            print(f"  Mean (μ): {stats['mean']:.2f}")
            print(f"  Standard Deviation (σ): {stats['std_dev']:.2f}")
            print(f"  Signal-to-Noise Ratio: {stats['snr']:.2f}")
            
        print("\nComparative Analysis:")
        print("--------------------")
        print("1. Field of View Comparison:")
        print(f"   - Main Camera FOV: {main_fov:.1f}° (wider angle)")
        print(f"   - Telephoto FOV: {tele_fov:.1f}° (narrower angle)")
        print(f"   - FOV Ratio: {main_fov/tele_fov:.1f}x (matches the optical zoom factor)")
        
        print("\n2. Noise Performance:")
        print("   The telephoto typically shows higher noise due to:")
        print("   - Smaller aperture (f/2.8 vs f/1.78)")
        print("   - Less light collection capability")
        print("   - Higher effective magnification")
        
        # Display images with analysis regions
        plt.figure(figsize=(15, 6))
        
        plt.subplot(121)
        plt.imshow(main_image)
        if analysis_region:
            x, y, w, h = analysis_region
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, 
                                            fill=False, color='red', linewidth=2))
        plt.title(f"Main Camera\nFOV: {main_fov:.1f}°")
        
        plt.subplot(122)
        plt.imshow(tele_image)
        if analysis_region:
            x, y, w, h = analysis_region
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, 
                                            fill=False, color='red', linewidth=2))
        plt.title(f"Telephoto Camera\nFOV: {tele_fov:.1f}°")
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find one or both image files.")
    except Exception as e:
        print(f"Error processing images: {e}")

def main():
    """
    Analysis of iPhone 15 Pro Main and Telephoto cameras.
    
    Technical Specifications:
    Main Camera (RCAM):
    - 48 MP main camera
    - 24mm focal length
    - f/1.78 aperture
    - 6.86mm sensor width
    
    Telephoto Camera (TCAM):
    - 12 MP
    - 77mm focal length (3x optical zoom)
    - f/2.8 aperture
    - 6.86mm sensor width
    """
    main_camera_specs = {
        'focal_length': 24,  # mm (24mm equivalent)
        'sensor_width': 6.86,  # mm
        'file': 'RCAM.dng'
    }
    
    tele_camera_specs = {
        'focal_length': 77,  # mm (77mm equivalent, 3x zoom)
        'sensor_width': 6.86,  # mm
        'file': 'TCAM.dng'
    }
    
    # Define region for noise analysis (adjust based on your images)
    analysis_region = (100, 100, 50, 50)  # (x, y, width, height)
    
    analyze_camera_comp(
        main_camera_specs['file'],
        tele_camera_specs['file'],
        main_camera_specs['focal_length'],
        tele_camera_specs['focal_length'],
        main_camera_specs['sensor_width'],
        tele_camera_specs['sensor_width'],
        analysis_region
    )

if __name__ == '__main__':
    main()
