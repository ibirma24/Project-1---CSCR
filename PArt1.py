import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
        region = img[h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2].copy()
        
        print(f"Analyzed 100x100 region statistics:")
        print(f"Mean (μ): {np.mean(region):.2f}")
        print(f"Standard deviation (σ): {np.std(region):.2f}\n")
        
        return {
            'filename': file_path.split('/')[-1],
            'mean': np.mean(region),
            'std': np.std(region),
            'min': np.min(region),
            'max': np.max(region),
            'patch': region,  # Store the actual region data
            'histogram': np.histogram(region, bins=50, range=(0, 255))
        }

def plot_analysis_results(dark_frames_results):
    """Plot analysis of dark frames."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Colors for different frames
    colors = ['navy', 'blue', 'royalblue']
    
    # Plot histograms (top subplot)
    for i, r in enumerate(dark_frames_results):
        hist, bins = r['histogram']
        ax1.plot(bins[:-1], hist, color=colors[i], alpha=0.6,
                label=f"{r['filename']}\nμ={r['mean']:.1f}, σ={r['std']:.1f}")
    
    ax1.set_title('Dark Frame Intensity Distribution', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    
    # Create violin plot (bottom subplot)
    positions = np.arange(len(dark_frames_results))
    data = [r['patch'].flatten() for r in dark_frames_results]
    labels = [f"Dark {i+1}" for i in range(len(dark_frames_results))]
    
    violin_parts = ax2.violinplot(data, positions=positions, widths=0.8)
    
    # Customize violin plots
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Customize violin lines
    for partname in ['cbars', 'cmins', 'cmaxes']:
        violin_parts[partname].set_edgecolor('black')
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_title('Dark Frame Distribution Comparison', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Pixel Value')
    
    plt.suptitle('Dark Frame Noise Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()

def print_analysis_summary(dark_frames_results):
    """Print noise analysis summary with clear tables and explanation."""
    print("\n=== Dark Frame Noise Analysis ===")
    print("+" + "-" * 52 + "+")
    print("|{:^52}|".format("Dark Frame Noise Measurements"))
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 11 + "+")
    print("|{:^15}|{:^12}|{:^12}|{:^11}|".format("Filename", "Mean (μ)", "Std Dev (σ)", "SNR"))
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 11 + "+")
    
    means = []
    stds = []
    for r in dark_frames_results:
        snr = r['mean'] / r['std']
        means.append(r['mean'])
        stds.append(r['std'])
        print("|{:<15}|{:>12.2f}|{:>12.2f}|{:>11.2f}|".format(
            r['filename'], r['mean'], r['std'], snr))
    
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 11 + "+")
    print("|{:<15}|{:>12.2f}|{:>12.2f}|{:>11.2f}|".format(
        "Average", np.mean(means), np.mean(stds), 
        np.mean(means)/np.mean(stds)))
    print("+" + "-" * 52 + "+")

    
    # Calculate noise metrics
    means = [r['mean'] for r in dark_frames_results]
    stds = [r['std'] for r in dark_frames_results]
    
    print("\nNoise Statistics Summary:")
    print("====================")
    print(f"Average Dark Current: {np.mean(means):.2f}")
    print(f"Average Noise Level (σ): {np.mean(stds):.2f}")
    print(f"Average SNR: {np.mean(means)/np.mean(stds):.2f}")

def main():
    # Define dark frames
    dark_frames = [Path(f) for f in ['IMG_7845.dng', 'IMG_7907.dng', 'IMG_7908.dng']]
    
    # Verify files exist
    dark_frames = [f for f in dark_frames if f.exists()]
    
    if not dark_frames:
        print("\nWarning: No dark frame files found!")
        print("Dark frames should be taken with the lens covered (2-3 second exposure)")
        return
    
    try:
        # Analyze dark frames
        print("\nAnalyzing dark frames...")
        dark_results = [analyze_raw_image(str(f)) for f in dark_frames]
        
        # Display results
        plot_analysis_results(dark_results)
        print_analysis_summary(dark_results)
        plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == '__main__':
    main()