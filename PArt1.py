import rawpy  
import numpy as np  
import matplotlib.pyplot as plt  
from pathlib import Path  

# Set up individual paths for each image
dark_image_path = Path("/Users/mateenibirogba/Documents/Computer Science/Project-1---CSCR/IMG_7845.dng")
light_image_path = Path("/Users/mateenibirogba/Documents/Computer Science/Project-1---CSCR/IMG_7846.dng")

# Create lists for dark and light frames
dark_frame_files = [dark_image_path]  # Dark frame (lens covered)
light_frame_files = [light_image_path]  # Light frame (white screen)

def analyze_noise(dng_files, patch_size=100, title="Noise Analysis"):
    """
    Analyze noise characteristics in the DNG files.
    Args:
        dng_files: List of paths to DNG files
        patch_size: Size of the square patch to analyze (default: 100)
        title: Title for the analysis plots (default: "Noise Analysis")
    Returns:
        List of dictionaries containing analysis results for each image
    """
    results = []

    for image_path in dng_files:
        with rawpy.imread(str(image_path)) as raw:
            raw_image = raw.raw_image_visible.copy()
            
            # Print basic information about the raw image
            print(f"\nRaw Image Information for {image_path.name}:")
            print(f"Shape: {raw_image.shape}")
            print(f"Data Type: {raw_image.dtype}")
            print(f"Min Value: {raw_image.min()}")
            print(f"Max Value: {raw_image.max()}\n")

        # Get dimensions
        if len(raw_image.shape) > 2:
            height, width, _ = raw_image.shape
        else:
            height, width = raw_image.shape

        # Center patch for analysis
        center_y, center_x = height // 2, width // 2
        cpatch = raw_image[center_y - patch_size // 2:center_y + patch_size // 2,
                          center_x - patch_size // 2:center_x + patch_size // 2]

        # Calculate the stats
        mean_stat = np.mean(cpatch)
        std_stat = np.std(cpatch)
        
        # Store results in a more accessible dictionary format
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

        # Plot detailed histogram
        plt.figure(figsize=(12, 6))
        
        # Calculate optimal number of bins using Sturges' rule
        n_bins = int(np.log2(len(cpatch.flatten())) + 1)
        
        # Plot histogram with more detail
        plt.hist(cpatch.flatten(), bins=n_bins, color='blue', alpha=0.7, density=True)
        plt.title(f"Pixel Intensity Distribution - {image_path.name}\nμ={mean_stat:.2f}, σ={std_stat:.2f}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        
        # Add mean and std dev lines
        plt.axvline(mean_stat, color='red', linestyle='dashed', alpha=0.8, label='Mean (μ)')
        plt.axvline(mean_stat + std_stat, color='green', linestyle=':', alpha=0.8, label='μ ± σ')
        plt.axvline(mean_stat - std_stat, color='green', linestyle=':', alpha=0.8)
        plt.legend()
        
        plt.show()

    return results 

# Image paths for dark and light frames
dark_frame_files = [
    Path("/Users/mateenibirogba/Documents/Computer Science/Project-1---CSCR/IMG_7845.dng")  # Dark frame (lens covered)
]

light_frame_files = [
    Path("/Users/mateenibirogba/Documents/Computer Science/Project-1---CSCR/IMG_7846.dng")  # Light frame (white screen)
]

def results_plot(dark_results, light_results):
    """
    Plot comparison between dark and bright frame analysis results.
    Args:
        dark_results: Results from dark frame analysis
        light_results: Results from bright frame analysis
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Dark frames analysis
    if dark_results:
        d_means = [r['mean'] for r in dark_results]
        d_stds = [r['std'] for r in dark_results]
        
        axes[0, 0].bar(range(len(dark_results)), d_stds)
        axes[0, 0].set_title('Dark Frame Standard Deviations')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Noise Std Dev')
        
        dark_patch = dark_results[0]['patch']
        axes[1, 0].hist(dark_patch.flatten(), bins=50, color='black', alpha=0.7)
        axes[1, 0].set_title('Dark Frame Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

    # Light frames analysis
    if light_results:
        l_means = [r['mean'] for r in light_results]
        l_stds = [r['std'] for r in light_results]
        
        axes[0, 1].bar(range(len(light_results)), l_stds)
        axes[0, 1].set_title('Light Frame Standard Deviations')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Noise Std Dev')
        
        light_patch = light_results[0]['patch']
        axes[1, 1].hist(light_patch.flatten(), bins=50, color='gray', alpha=0.7)
        axes[1, 1].set_title('Light Frame Histogram')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

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