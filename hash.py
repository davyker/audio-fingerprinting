import librosa
from skimage.feature import peak_local_max
import numpy as np
import os
from visualisation import plot_spectrogram_and_peaks 

def downsample_audio(y, sr, target_sr=22050):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def process_audio(filename, sample_rate=22050, n_fft=1024, window='hann', win_length=1024, hop_length=128, args=None):
    """
    Process an audio file to extract its waveform and spectrograms.
    
    Args:
        filename: Path to the audio file
        sample_rate: Target sample rate for audio processing
        n_fft: FFT size
        window: Window type for STFT
        win_length: Window length for STFT
        hop_length: Hop length for STFT
        args: Configuration arguments from config.yaml containing masking parameters
        
    Returns:
        Tuple (audio_data, sample_rate, spectrogram, log_spectrogram)
    """
    y, sr = librosa.load(os.path.join(filename))

    y, sr = downsample_audio(y, sr, target_sr=sample_rate)

    # Compute STFT spectrogram
    D = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length))
    
    # Apply frequency mask if specified in args
    if hasattr(args, 'apply_mask') and args.apply_mask:
        freq_bins = np.linspace(0, sample_rate/2, D.shape[0])
        range_to_boost = (freq_bins >= args.boost_band_min) & (freq_bins <= args.boost_band_max)
        D[range_to_boost, :] *= args.boost_ratio
        D[~range_to_boost, :] *= args.suppress_ratio
    
    # Apply max norm
    if np.max(D) > 0:
        D = D / np.max(D)
    
    D_log = np.log(D + 1e-10)

    return y, sr, D, D_log

def detect_peaks(spectrogram, min_distance=15, max_peaks=np.inf, threshold_rel=None):
    """
    Detect peaks in a spectrogram.
    
    Args:
        spectrogram: The spectrogram to detect peaks in
        min_distance: Minimum distance between peaks
        max_peaks: Maximum number of peaks to detect
        threshold_rel: Relative threshold for peak detection
        
    Returns:
        Array of peak coordinates
    """
    peak_coords = peak_local_max(spectrogram, min_distance=min_distance, 
                                threshold_rel=threshold_rel, num_peaks=max_peaks)
    
    return peak_coords

def bin_frequencies(coordinates, n_bins, n_fft=1024):
    """
    Bin the frequency column of the list of coordinates of peaks.
    
    Args:
        coordinates: Array of coordinates to bin
        n_bins: Number of bins to use
        n_fft: FFT size used in spectrogram computation
    """
    coordinates_binned = coordinates.copy()
    coordinates_binned[:, 0] = np.digitize(coordinates[:, 0], np.linspace(0, n_fft/2, n_bins+1))-1
    
    return coordinates_binned

def create_target_zone_boundaries(anchor_point, width=300, height=200, forward_offset=2, upward_offset=0):
    """
    Create target zone boundaries based on the anchor point.
    
    Args:
        anchor_point: The anchor point [freq_bin, time_frame]
        width: Width of the target zone in time frames
        height: Height of the target zone in frequency bins
        forward_offset: Offset from anchor point in time frames
        upward_offset: Vertical offset from anchor point
    
    Returns:
        Tuple of (min_freq, max_freq, min_time, max_time)
    """
    # The default position, with offsets at 0 is for the target zone to be a rectangle with
    #       anchor point in the middle of the left edge
    anchor_freq, anchor_time = anchor_point

    min_time = anchor_time + forward_offset
    max_time = min_time + width

    min_freq = max(0, anchor_freq - height//2 + upward_offset)
    max_freq = anchor_freq + height//2 + upward_offset

    return min_freq, max_freq, min_time, max_time

def get_target_points(anchor_point, peaks, spectrogram=None, width=300, height=200, 
                  forward_offset=2, upward_offset=0, target_selection_mode='highest', max_points=100):
    """
    Create a target zone for a given anchor point and select up to max_points with highest amplitude.
    
    Args:
        anchor_point: The anchor point [freq_bin, time_frame]
        peaks: Array of peak points
        spectrogram: Spectrogram data to get amplitude values (log scale)
        width: Width of the target zone in time frames
        height: Height of the target zone in frequency bins
        forward_offset: Offset from anchor point in time frames
        upward_offset: Vertical offset from anchor point
        target_selection_mode: How to select target points ('highest' or 'random')
        max_points: Maximum number of target points to return
    
    Returns:
        Array of peak points that are in the target zone
    """
    # Get target zone boundaries
    anchor_freq, anchor_time = anchor_point
    min_freq, max_freq, min_time, max_time = create_target_zone_boundaries(
        anchor_point, width=width, height=height, forward_offset=forward_offset, upward_offset=upward_offset
    )
    
    # Vectorised filtering of points in target zone - this is enormously faster than looping through every peak
    mask = ((peaks[:, 1] >= min_time) & (peaks[:, 1] <= max_time) & 
            (peaks[:, 0] >= min_freq) & (peaks[:, 0] <= max_freq))
    
    # Remove the anchor point itself if present
    if min_freq <= anchor_freq <= max_freq and min_time <= anchor_time <= max_time:
        mask = mask & ~((peaks[:, 0] == anchor_freq) & (peaks[:, 1] == anchor_time))
    target_points = peaks[mask]
    
    if len(target_points) == 0:
        return np.empty((0, 2), dtype=int)

    if len(target_points) > max_points:
        if target_selection_mode == 'highest' and spectrogram is not None:
            # Vectorised amplitude collection
            freq_indices = target_points[:, 0]
            time_indices = target_points[:, 1]
            amplitudes = spectrogram[freq_indices, time_indices]
            
            # Get indices of the max_points highest amplitude points
            top_indices = np.argsort(amplitudes)[-max_points:]
            target_points = target_points[top_indices]
        elif target_selection_mode == 'closest':
            # Find the closest points to the anchor point
            distances = np.linalg.norm(target_points - anchor_point, axis=1)
            closest_indices = np.argsort(distances)[:max_points]
            target_points = target_points[closest_indices]
        else:
            if target_selection_mode == 'highest':
                # In this case, no spectrogram was provided on which to base the selection of highest points
                print("Warning: No spectrogram provided, using random selection instead.")
            indices = np.random.choice(len(target_points), max_points, replace=False)
            target_points = target_points[indices]
    
    return target_points

def generate_hash(anchor_point, target_point):
    """
    Generate a hash from an anchor point and a target point.
    
    Hash format: (anchor_freq, target_freq, time_delta)
    """
    anchor_freq, anchor_time = anchor_point
    target_freq, target_time = target_point
    time_delta = target_time - anchor_time

    return (anchor_freq, target_freq, time_delta)

def create_fingerprint(peaks, spectrogram=None, width=300, height=200,
                       forward_offset=2, max_target_points=100, target_selection_mode='highest', args=None):
    """
    Create a fingerprint from a set of peaks.
    
    Args:
        peaks: Array of peak points
        spectrogram: Spectrogram data to get amplitude values
        width: Width of the target zone
        height: Height of the target zone
        forward_offset: Offset from anchor point
        max_target_points: Maximum number of target points per anchor
        target_selection_mode: Method for selecting target points ('highest' or 'random')
        args: Configuration arguments containing hash_mode and pairs_implies_points settings
    
    Returns:
        Dictionary of hashes to (anchor_time, [matching_target_times])
    """
    fingerprint = {}
    print(f"     Creating fingerprint from {len(peaks)} peaks...")
    
    # Get hash mode from args if provided, otherwise use default 'pairs'
    hash_mode = args.hash_mode if args and hasattr(args, 'hash_mode') else 'pairs'
    pairs_implies_points = args.pairs_implies_points if args and hasattr(args, 'pairs_implies_points') else False
    
    for anchor_idx, anchor_point in enumerate(peaks):
        # Get points in target zone
        if hash_mode == 'pairs':
            target_points = get_target_points(
                anchor_point, 
                peaks, 
                spectrogram=spectrogram,
                width=width, 
                height=height, 
                forward_offset=forward_offset,
                target_selection_mode=target_selection_mode,
                max_points=max_target_points
            )
            
            if len(target_points) == 0:
                continue
            
            target_points = target_points.tolist()
            if pairs_implies_points:
                target_points.append(anchor_point)  # If specified, include the anchor point itself

            anchor_time = anchor_point[1]
            
            # Generate all hash values for this anchor
            hash_values = [generate_hash(anchor_point, tp) for tp in target_points]
            # Update the fingerprint dictionary
            for hash_value in hash_values:
                fingerprint.setdefault(hash_value, []).append(anchor_time)
        elif hash_mode == 'points':
            hash_value = generate_hash(anchor_point, anchor_point)
            anchor_time = anchor_point[1]
            if hash_value in fingerprint:
                fingerprint[hash_value].append(anchor_time)
            else:
                fingerprint[hash_value] = [anchor_time]
    return fingerprint


def process_file_for_fingerprinting(file_path, args, plot=False, is_query=None, min_distance=None, max_peaks=None):
    """
    Process an audio file and prepare it for fingerprinting.
    
    Args:
        file_path: Path to the audio file
        args: Command line arguments with configuration
        plot: Whether to show plots
        is_query: Whether this is a query file. If None, will be determined from the path.
        min_distance: Min distance for peak detection. If None, will use appropriate value based on file type.
        max_peaks: Maximum number of peaks to detect. If None, will use appropriate value based on file type.
        
    Returns:
        Dictionary with processed audio data
    """
    # Determine if is a query file if not explicitly specified
    if is_query is None:
        # Check if file path contains query directory or has snippet in the name
        is_query = args.query_dir in file_path or 'snippet' in file_path
    
    # Process audio file
    y, sr, D, D_log = process_audio(
        file_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        window=args.window,
        win_length=args.win_length,
        hop_length=args.hop_length,
        args=args
    )
    
    # Determine appropriate min_distance and max_peaks values based on is_query
    if min_distance is None:
        min_distance = args.query_min_distance if is_query else args.db_min_distance
        
    if max_peaks is None:
        max_peaks = args.query_max_peaks if is_query else args.db_max_peaks
    
    # Get peaks from log spectrogram
    peak_coords = detect_peaks(
        D_log,
        min_distance=min_distance,
        max_peaks=max_peaks,
        threshold_rel=args.threshold_rel
    )
    
    # Calculate the actual number of bins based on peaks_f_bins
    # If between 0 and 1 (inclusive), it is a ratio by which to reduce the number of bins (1 means no binning)
    # If int and >=2, it is the number of bins to use
    if 0 < args.peaks_f_bins <= 1:
        actual_peaks_f_bins = int(args.peaks_f_bins * D.shape[0])
    elif args.peaks_f_bins > 1 and isinstance(args.peaks_f_bins, int):
        actual_peaks_f_bins = args.peaks_f_bins
    else:
        raise ValueError("peaks_f_bins must be a positive integer or a float between 0 and 1")
    
    # Bin frequency values
    peak_coords_binned = bin_frequencies(peak_coords, n_bins=actual_peaks_f_bins, n_fft=args.n_fft)
    
    # Choose target selection mode based on file type
    target_selection_mode = args.query_hash_selection_mode if is_query else args.db_hash_selection_mode
    
    # Create fingerprint
    fingerprint = create_fingerprint(
        peak_coords_binned, 
        spectrogram=D_log, 
        width=args.target_zone_width,
        height=args.target_zone_height,
        forward_offset=args.forward_offset,
        max_target_points=args.max_target_points,
        target_selection_mode=target_selection_mode,
        args=args
    )
    
    peak_coords_scaled = None
    if plot:
        peak_coords_scaled = peak_coords_binned.copy()
        peak_coords_scaled[:, 0] = peak_coords_scaled[:, 0] * (D.shape[0] / actual_peaks_f_bins)
        
        plot_spectrogram_and_peaks(
            D_log,
            peak_coords_scaled,
            sr=sr,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            title=f"Spectrogram with Peaks (min_distance={min_distance}, max_peaks={max_peaks})"
        )
        
    # Determine the min_distance and max_peaks values used, based on is_query
    min_distance_used = min_distance if min_distance is not None else (args.query_min_distance if is_query else args.db_min_distance)
    max_peaks_used = max_peaks if max_peaks is not None else (args.query_max_peaks if is_query else args.db_max_peaks)
        
    return {
        'audio': y,
        'sample_rate': sr,
        'spectrogram': D,
        'log_spectrogram': D_log,
        'peak_coords': peak_coords,
        'binned_peaks': peak_coords_binned,
        'scaled_peaks': peak_coords_scaled,
        'fingerprint': fingerprint,
        'min_distance_used': min_distance_used,
        'max_peaks_used': max_peaks_used
    }

def get_matching_database_file(query_file, database_dir="database_recordings"):
    """
    Find the matching database file for a given query file.
    
    Args:
        query_file: Path to the query file
        database_dir: Directory containing database files
        
    Returns:
        Path to the matching database file, or None if not found
    """
    # Get the query file basename (e.g., "pop.00034-snippet-10-0.wav")
    query_basename = os.path.basename(query_file)
    
    # Extract the prefix (e.g., "pop.00034")
    prefix = query_basename.split('-snippet')[0]
    
    # Construct the expected database filename
    db_filename = f"{prefix}.wav"
    db_filepath = os.path.join(database_dir, db_filename)
    
    # Check if the database file exists
    if os.path.exists(db_filepath):
        return db_filepath
    else:
        return None