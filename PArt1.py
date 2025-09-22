import rawpy
import numpy as np
import matplotlib.pyplot as plt

def analyze_raw_image(file_path):
    """Analyze noise in a DNG image's center region."""
    with rawpy.imread(file_path) as raw:
        # Get raw data and print basic information
        raw_data = raw.raw_image
        print(f"\nRaw image information for {file_path}:")
        print(f"Shape: {raw_data.shape}")
        print(f"Data type: {raw_data.dtype}")
        print(f"Min value: {raw_data.min()}")
        print(f"Max value: {raw_data.max()}\n")
        
        # Get processed grayscale data
        img = np.mean(raw.postprocess(), axis=2) if raw.postprocess().ndim == 3 else raw.postprocess()
        
        # Analyze center 100x100 region
        h, w = img.shape
        size = min(100, h//2, w//2)
        region = img[h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]
        
        print(f"Analyzed 100x100 region statistics:")
        print(f"Mean (μ): {np.mean(region):.2f}")
        print(f"Standard deviation (σ): {np.std(region):.2f}\n")
        
        return {
            'filename': file_path.split('/')[-1],
            'mean': np.mean(region),
            'std': np.std(region),
            'min': np.min(region),
            'max': np.max(region),
            'histogram': np.histogram(region, bins=50, range=(0, 255))
        }

def plot_analysis_results(dark_frames_results, bright_frames_results):
    """Plot analysis of dark and bright frames."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    colors = ['b', 'r', 'g']
    
    # Plot dark and bright frame histograms
    for i, (result, title) in enumerate([(dark_frames_results, 'Dark Frames'),
                                       (bright_frames_results, 'Bright Frames')]):
        ax = ax1 if i == 0 else ax2
        for j, r in enumerate(result):
            hist, bins = r['histogram']
            ax.plot(bins[:-1], hist, color=colors[j], alpha=0.7,
                   label=f"{r['filename']}\nμ={r['mean']:.1f}, σ={r['std']:.1f}")
        ax.set_title(f'{title} Intensity Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
    
   
 
    plt.suptitle('Noise Analysis: Dark Frames vs Bright Frames', fontsize=14)
    plt.tight_layout()
    plt.show()

def print_analysis_summary(dark_frames_results, bright_frames_results):
    """Print noise analysis summary with clear tables and explanation."""
    def print_frame_table(results, frame_type):
        means = [r['mean'] for r in results]
        stds = [r['std'] for r in results]
        
        print(f"\n{frame_type} Frames Analysis")
        print("-" * 50)
        print(f"{'Filename':<15} {'Mean (μ)':<12} {'Std Dev (σ)':<12} {'SNR':<8}")
        print("-" * 50)
        for r in results:
            snr = r['mean'] / r['std']
            print(f"{r['filename']:<15} {r['mean']:8.2f}    {r['std']:8.2f}    {snr:6.2f}")
        print("-" * 50)
        print(f"Average:      {np.mean(means):8.2f}    {np.mean(stds):8.2f}    {np.mean(means)/np.mean(stds):6.2f}")
        return means, stds
    
    print("\nNoise Analysis Results")
    print("====================")
    dark_means, dark_stds = print_frame_table(dark_frames_results, "Dark")
    bright_means, bright_stds = print_frame_table(bright_frames_results, "Bright")
    
    print("\nNoise Comparison:")
    print(f"Shot Noise Contribution: {np.mean(bright_stds) - np.mean(dark_stds):.2f}")
    print(f"SNR Improvement: {(np.mean(bright_means)/np.mean(bright_stds))/(np.mean(dark_means)/np.mean(dark_stds)):.2f}x")

def main():
    # Define image sets
    dark_frames = ['IMG_7845.dng', 'IMG_7907.dng', 'IMG_7908.dng']
    bright_frames = ['IMG_7846.dng', 'IMG_7909.dng', 'IMG_7910.dng']
    
    # Analyze images
    try:
        dark_results = [analyze_raw_image(f) for f in dark_frames]
        bright_results = [analyze_raw_image(f) for f in bright_frames]
        
        # Display results
        plot_analysis_results(dark_results, bright_results)
        print_analysis_summary(dark_results, bright_results)
        plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
import numpy as np  
import matplotlib.pyplot as plt  
from pathlib import Path  

# Set up individual paths for each image
# Use relative paths for portability
dark_image_path = Path("IMG_7845.dng")
light_image_path = Path("IMG_7846.dng")

# Create lists for dark and light frames
dark_frame_files = [dark_image_path]  # Dark frame (lens covered)
light_frame_files = [light_image_path]  # Light frame (white screen)

def analyze_noise(dng_files, patch_size=100, title="Noise Analysis"):
    """
    Analyze noise characteristics in the DNG files.
    """
    results = []
    # Create a single figure for all histograms
    plt.figure(figsize=(15, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

    for idx, image_path in enumerate(dng_files):
        with rawpy.imread(str(image_path)) as raw:
            raw_image = raw.raw_image_visible.copy()
            
            # Print basic information about the raw image
            print(f"\nRaw Image Information for {image_path.name}:")
            print(f"Shape: {raw_image.shape}")
            print(f"Data Type: {raw_image.dtype}")
            print(f"Min Value: {raw_image.min()}")
            print(f"Max Value: {raw_image.max()}\n")

        # Get dimensions and extract center patch
        height, width = raw_image.shape[:2]
        center_y, center_x = height // 2, width // 2
        cpatch = raw_image[center_y - patch_size // 2:center_y + patch_size // 2,
                          center_x - patch_size // 2:center_x + patch_size // 2]

        # Calculate stats
        mean_stat = np.mean(cpatch)
        std_stat = np.std(cpatch)
        
        # Store results
        result_dict = {
            'filename': image_path.name,
            'mean': mean_stat,
            'std': std_stat,
            'patch': cpatch
        }
        results.append(result_dict)

        # Print results
        print(f"File: {image_path.name}")
        print(f"Mean: {mean_stat:.2f}")
        print(f"Standard Deviation: {std_stat:.2f}\n")

        # Plot histogram in shared figure
        n_bins = int(np.log2(len(cpatch.flatten())) + 1)
        plt.hist(cpatch.flatten(), bins=n_bins, color=colors[idx % len(colors)], 
                alpha=0.5, density=True, label=f"{image_path.name}\nμ={mean_stat:.2f}, σ={std_stat:.2f}")

    # Customize shared plot
    plt.title("Pixel Intensity Distribution - All Images", fontsize=14, pad=20)
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return results
# Image paths for dark and light frames
# (Removed duplicate definitions of dark_frame_files and light_frame_files)


def results_plot(dark_results, light_results):
    """Plot comparative analysis of dark and light frames."""
    fig = plt.figure(figsize=(15, 6))
    
    # Standard deviations comparison
    plt.subplot(1, 2, 1)
    d_stds = [r['std'] for r in dark_results]
    l_stds = [r['std'] for r in light_results]
    
    x = np.arange(max(len(dark_results), len(light_results)))
    width = 0.35
    
    plt.bar(x - width/2, d_stds, width, label='Dark Frames', color='darkblue', alpha=0.7)
    plt.bar(x + width/2, l_stds, width, label='Light Frames', color='orange', alpha=0.7)
    
    plt.title('Standard Deviations Comparison')
    plt.xlabel('Image Index')
    plt.ylabel('Noise (Standard Deviation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Means comparison
    plt.subplot(1, 2, 2)
    d_means = [r['mean'] for r in dark_results]
    l_means = [r['mean'] for r in light_results]
    
    plt.bar(x - width/2, d_means, width, label='Dark Frames', color='darkblue', alpha=0.7)
    plt.bar(x + width/2, l_means, width, label='Light Frames', color='orange', alpha=0.7)
    
    plt.title('Mean Intensity Comparison')
    plt.xlabel('Image Index')
    plt.ylabel('Mean Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
# Main execution
if __name__ == "__main__":
    # Check if the image files exist
    dark_frames_exist = any(path.exists() for path in dark_frame_files)
    light_frames_exist = any(path.exists() for path in light_frame_files)

    if not dark_frames_exist:
        print("\nWarning: No dark frame files found!")
        print("Please add your dark frame DNG files to the project directory")
        print("Dark frames should be taken with the lens covered")
        dark_results = []
    else:
        print("\nAnalyzing dark frames...")
        dark_results = analyze_noise(dark_frame_files)

    if not light_frames_exist:
        print("\nWarning: No light frame files found!")
        print("Please add your light frame DNG files to the project directory")
        print("Light frames should be taken of a uniformly lit white surface")
        light_results = []
    else:
        print("\nAnalyzing light frames...")
        light_results = analyze_noise(light_frame_files)

    # Plot comparative results if we have both types of frames
    if dark_results and light_results:
        results_plot(dark_results, light_results)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 20)
        print(f"Dark frames average std: {np.mean([r['std'] for r in dark_results]):.2f}")
        print(f"Light frames average std: {np.mean([r['std'] for r in light_results]):.2f}")
    else:
        print("\nCannot generate comparison: Need both dark and light frames for analysis")