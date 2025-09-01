import numpy as np
import matplotlib.pyplot as plt
import os

def plot_spectrogram_and_peaks(spectrogram, peaks, sr=22050, hop_length=128, n_fft=1024, title="Spectrogram with Peaks"):
    """
    Plot a spectrogram with identified peaks highlighted.
    
    Args:
        spectrogram: The spectrogram to display
        peaks: Array of peak coordinates [freq, time]
        sr: Sample rate
        hop_length: Hop length used to generate the spectrogram
        n_fft: FFT size
        title: Plot title
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    
    # Calculate time and frequency values
    time_frames = np.arange(spectrogram.shape[1])
    freq_bins = np.arange(spectrogram.shape[0])
    
    # Convert frames to seconds
    time_seconds = time_frames * hop_length / sr
    
    # Convert bins to Hz
    fft_freqs = np.linspace(0, sr/2, n_fft//2 + 1)
    
    # Plot 1: Original spectrogram (left side)
    im1 = ax1.imshow(spectrogram, aspect='auto', origin='lower')
    ax1.set_title("Original Spectrogram")
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Frequency bins')
    fig.colorbar(im1, ax=ax1)
    
    # Create a second x-axis on top with time in seconds
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks([0, spectrogram.shape[1]//4, spectrogram.shape[1]//2, 3*spectrogram.shape[1]//4, spectrogram.shape[1]-1])
    ax1_top.set_xticklabels([f"{t:.1f}" for t in time_seconds[[0, spectrogram.shape[1]//4, spectrogram.shape[1]//2, 
                                                            3*spectrogram.shape[1]//4, spectrogram.shape[1]-1]]])
    ax1_top.set_xlabel('Time (seconds)')
    
    # Create a second y-axis on right with frequency in Hz
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    max_bin = spectrogram.shape[0] - 1
    ax1_right.set_yticks([0, max_bin//4, max_bin//2, 3*max_bin//4, max_bin])
    ax1_right.set_yticklabels([f"{f/1000:.1f}" for f in fft_freqs[[0, max_bin//4, max_bin//2, 3*max_bin//4, max_bin]]])
    ax1_right.set_ylabel('Frequency (kHz)', labelpad=-15)
    
    # Plot 2: Spectrogram with peaks
    im2 = ax2.imshow(spectrogram, aspect='auto', origin='lower')
    ax2.set_title(title)
    ax2.set_xlabel('Time (frames)')
    fig.colorbar(im2, ax=ax2)
    
    # Create a second x-axis on top with time in seconds for plot 2
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks([0, spectrogram.shape[1]//4, spectrogram.shape[1]//2, 3*spectrogram.shape[1]//4, spectrogram.shape[1]-1])
    ax2_top.set_xticklabels([f"{t:.1f}" for t in time_seconds[[0, spectrogram.shape[1]//4, spectrogram.shape[1]//2, 
                                                            3*spectrogram.shape[1]//4, spectrogram.shape[1]-1]]])
    ax2_top.set_xlabel('Time (seconds)')
    
    # Create a second y-axis on right with frequency in Hz for plot 2
    ax2_right = ax2.twinx()
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.set_yticks([0, max_bin//4, max_bin//2, 3*max_bin//4, max_bin])
    ax2_right.set_yticklabels([f"{f/1000:.1f}" for f in fft_freqs[[0, max_bin//4, max_bin//2, 3*max_bin//4, max_bin]]])
    ax2_right.set_ylabel('Frequency (kHz)', labelpad=-15)  # Move label to the left with negative padding
    
    # Get the coordinates of the peaks and add them to the right plot
    print(f"Peak coordinates: {peaks[:5]}...\n{peaks[-5:]}, shape: {peaks.shape}")
    ax2.scatter(peaks[:,1], peaks[:,0], c='r', s=15, edgecolors='white', alpha=0.8, marker='o')
    
    plt.tight_layout()
    plt.show()

def plot_target_zone(spectrogram, peaks, anchor_idx=None, sr=22050, 
                 max_points=100, width=300, height=200, forward_offset=2, 
                 title="Spectrogram with Target Zone"):
    """
    Plot the spectrogram, peaks, and the target zone for a randomly selected anchor point.
    
    Args:
        spectrogram: The spectrogram to plot
        peaks: Array of peak points
        anchor_idx: Index of the anchor point to use, if None a random one will be selected
        sr: Sample rate
        max_points: Maximum number of target points to display
        width: Width of the target zone
        height: Height of the target zone
        forward_offset: Offset from anchor point in time frames
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.scatter(peaks[:, 1], peaks[:, 0], c='r', s=10, alpha=0.5)
    
    # If anchor_idx is not specified, randomly select one
    if anchor_idx is None:
        if len(peaks) > 0:
            anchor_idx = np.random.randint(0, len(peaks))
        else:
            plt.title("No peaks detected")
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency bins')
            plt.show()
            return
    
    anchor_point = peaks[anchor_idx]
    
    # Create target zone boundaries
    anchor_freq, anchor_time = anchor_point
    min_time = anchor_time + forward_offset
    max_time = min_time + width
    min_freq = max(0, anchor_freq - height//2)
    max_freq = anchor_freq + height//2
    
    # Find target zone points (all peaks in the zone except the anchor)
    mask = ((peaks[:, 1] >= min_time) & (peaks[:, 1] <= max_time) & 
            (peaks[:, 0] >= min_freq) & (peaks[:, 0] <= max_freq))
    mask = mask & ~((peaks[:, 0] == anchor_freq) & (peaks[:, 1] == anchor_time))
    target_zone = peaks[mask]
    
    # If too many target points, take a subset
    if len(target_zone) > max_points:
        indices = np.random.choice(len(target_zone), max_points, replace=False)
        target_zone = target_zone[indices]
    
    # Plot anchor point
    plt.scatter(anchor_point[1], anchor_point[0], c='g', s=100, marker='*', label='Anchor')
    
    # Plot target zone points
    if len(target_zone) > 0:
        plt.scatter(target_zone[:, 1], target_zone[:, 0], c='b', s=30, marker='o', 
                   label=f'Target Zone (max {max_points} points)')
    
    # Draw target zone boundary
    plt.plot([min_time, max_time, max_time, min_time, min_time], 
             [min_freq, min_freq, max_freq, max_freq, min_freq], 
             'w--', linewidth=1)
    
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency bins')
    plt.legend()
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def plot_match_histogram(match_result, title="Hash Match Histogram"):
    """
    Plot a histogram of the offsets from matching fingerprints.
    """
    if 'histogram' not in match_result or not match_result['histogram']:
        print("No matches to plot")
        return
    
    # Get offsets and their counts
    offsets = np.array(list(match_result['histogram'].keys()))
    counts = np.array(list(match_result['histogram'].values()))
    
    # Sort by offset
    sorted_indices = np.argsort(offsets)
    offsets = offsets[sorted_indices]
    counts = counts[sorted_indices]
    
    plt.figure(figsize=(10, 5))
    plt.bar(offsets, counts, width=1)
    plt.xlabel('Time Offset (frames)')
    plt.ylabel('Number of Matching Hashes')
    plt.title(f"{title} - Best Offset: {match_result['best_offset']} frames ({match_result['best_score']} matches)")
    
    if match_result['best_offset'] is not None:
        plt.axvline(match_result['best_offset'], color='r', linestyle='--', label=f"Best Offset: {match_result['best_offset']}")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_evaluation_results(df, save_path=None):
    """
    Plot evaluation results.
    
    Args:
        df: DataFrame with evaluation results
        save_path: Path to save the plot to. If None, the plot is only displayed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Plot 1: Bar chart of true match positions
    position_counts = df['true_match_position'].value_counts().sort_index()
    axes[0].bar(position_counts.index, position_counts.values)
    axes[0].set_xlabel('True Match Position')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of True Match Positions')
    
    if not position_counts.empty:
        max_position = int(position_counts.index.max())
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Plot 2: Scatter plot of true match score vs. top score with log-log scales
    # Add 1 to all values to handle zeros
    x_vals = df['true_match_score'] + 1
    y_vals = df['top_score'] + 1
    
    # Set up log-log axes
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    # Colour points by genre
    pop_df = df[df['genre'] == 'pop']
    classical_df = df[df['genre'] == 'classical']
    
    # Create the scatter plot colored by genre
    if not pop_df.empty:
        axes[1].scatter(pop_df['true_match_score'] + 1, pop_df['top_score'] + 1, 
                       color='blue', marker='o', alpha=0.7, label='Pop')
    if not classical_df.empty:
        axes[1].scatter(classical_df['true_match_score'] + 1, classical_df['top_score'] + 1, 
                       color='red', marker='s', alpha=0.7, label='Classical')
    
    # Add a diagonal reference line (y=x) on log-log scale
    if not df.empty:
        max_val = max(x_vals.max(), y_vals.max())
        min_val = 1  # Minimum value after adding 1
        axes[1].plot([min_val, max_val], [min_val, max_val], 'k--')  # Diagonal line
    
    # Function to format tick labels to show original values
    def format_ticks(x, _):
        # Subtract 1 to show original value
        return f"{int(x-1)}" if x > 1 else "0"
    
    from matplotlib.ticker import FuncFormatter
    axes[1].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axes[1].yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    axes[1].set_xlabel('True Match Score (log scale)')
    axes[1].set_ylabel('Top Score (log scale)')
    axes[1].set_title('True Match Score vs. Top Score by Genre (log-log)')
    axes[1].legend()
    
    # Add a grid
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Bar chart comparing genre accuracy
    if hasattr(df, 'attrs') and 'pop_accuracy' in df.attrs:
        genres = ['Overall', 'Pop', 'Classical']
        accuracies = [df.attrs['overall_accuracy'], df.attrs['pop_accuracy'], df.attrs['classical_accuracy']]
        counts = [
            f"{df.attrs['overall_accuracy']:.2f}\n({df.shape[0]} files)",
            f"{df.attrs['pop_accuracy']:.2f}\n({df.attrs['pop_total']} files)",
            f"{df.attrs['classical_accuracy']:.2f}\n({df.attrs['classical_total']} files)"
        ]
        
        bar_colors = ['gray', 'blue', 'red']
        bars = axes[2].bar(genres, accuracies, color=bar_colors)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    counts[i], ha='center', va='bottom')
        
        axes[2].set_ylim(0, 1.1) 
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Identification Accuracy by Genre')
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()