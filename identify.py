import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
from collections import defaultdict, Counter
import pickle
from multiprocessing import Pool, cpu_count
import gzip

from hash import *
from utils.yaml_config_hook import yaml_config_hook

def process_database_file(args_tuple):
    """Helper function for parallel processing of database files."""
    db_idx, db_file, args = args_tuple
    track_id = os.path.basename(db_file)
    
    print(f"Processing database file #{db_idx}/{args.total_files}: {track_id}...")
    
    # Process the database file
    db_file_data = process_file_for_fingerprinting(
        db_file, 
        args=args,
        plot=False, 
        is_query=False,
        min_distance=args.db_min_distance,
        max_peaks=args.db_max_peaks
    )
    
    total_hashes_this_track = sum(len(times) for times in db_file_data['fingerprint'].values())
    print(f"     Fingerprint contains {len(db_file_data['fingerprint'])} hashes and {total_hashes_this_track} hash-time pairs")
    
    return track_id, db_file_data['fingerprint']

def build_database_hash_map(database_files, args, use_parallel=True):
    """
    Build a combined hash map for all database files with an optimised structure.
    
    Args:
        database_files: List of database file paths (can be a subset of all database files)
        args: Command line arguments with configuration
        use_parallel: Whether to use parallel processing (default: True)
        
    Returns:
        hash map: a dictionry of dictionaries:
        {
            hash_value: {
                track_id1: [time_offset1, time_offset2, ...],
                track_id2: [time_offset1, time_offset2, ...],
                ...
            },
            ...
        }
    """
    if use_parallel and len(database_files) > 1:
        # Add total_files to args for progress display
        args.total_files = len(database_files)
        
        # Use parallel processing
        num_workers = min(cpu_count()-2, len(database_files))
        print(f"Using {num_workers} parallel workers to process {len(database_files)} files...")
        
        with Pool(num_workers) as pool:
            results = pool.map(process_database_file, 
                             [(idx, db_file, args) for idx, db_file in enumerate(database_files)])
        
        hash_map = {}
        print("Combining results into hash map...")
        for track_id, fingerprint in results:
            for hash_value, times in fingerprint.items():
                if hash_value not in hash_map:
                    hash_map[hash_value] = {track_id: times}
                elif track_id not in hash_map[hash_value]:
                    hash_map[hash_value][track_id] = times
                else:
                    hash_map[hash_value][track_id].extend(times)
    else:
        # Sequential processing
        hash_map = defaultdict(lambda: defaultdict(list))
        total_hash_pairs = 0
        for db_idx, db_file in enumerate(database_files):
            # Get track ID (filename without path and extension)
            track_id = os.path.basename(db_file)
            
            # Process and fingerprint the database file
            print(f"Processing database file #{db_idx}/{len(database_files)}: {track_id}...")
            
            # Process the database file
            db_file_data = process_file_for_fingerprinting(
                db_file, 
                args=args,
                plot=False, 
                is_query=False,
                min_distance=args.db_min_distance,
                max_peaks=args.db_max_peaks
            )
            
            total_hashes_this_track = 0
            # Iterate over all hashes in the fingerprint
            for hash_value, times in db_file_data['fingerprint'].items():
                total_hashes_this_track += len(times)
                # add all offsets for this track and hash value
                hash_map[hash_value][track_id].extend(times)
            print(f"     Fingerprint contains {len(db_file_data['fingerprint'])} hashes and {total_hashes_this_track} hash-time pairs")
            total_hash_pairs += total_hashes_this_track

    total_hash_pairs = sum(len(times) for track_dict in hash_map.values() for times in track_dict.values())
    print(f"Database hash map built: {len(hash_map)} unique hashes for {len(database_files)} tracks")
    print(f"Total hash-time pairs: {total_hash_pairs}")
    return hash_map

def identify_query(query_file, hash_map, args, plot=False):
    """
    Identify which track in the database best matches the query using the hash map.
    
    Args:
        query_file: Path to the query audio file
        hash_map: Combined hash map of all database files
        args: Command line arguments with configuration
        plot: Whether to show plots
    
    Returns:
        A list of (track_id, score, offset_seconds, confidence) tuples, sorted by score or confidence
    """
    # Process query file
    query_data = process_file_for_fingerprinting(
        query_file, 
        args=args,
        plot=plot,
        is_query=True,
        min_distance=args.query_min_distance,
        max_peaks=args.query_max_peaks
    )
    query_fingerprint = query_data['fingerprint']
    
    # Print number of hashes
    true_total_hashes = 0
    for hash_value, times in query_fingerprint.items():
        true_total_hashes += len(times)
    print(f"Query fingerprint contains {len(query_fingerprint)} hashes and {true_total_hashes} hash-time pairs")
    
    # Track matches and offsets
    matches = defaultdict(list)
    
    # Match query hashes against the hash map
    for hash_value, query_times in query_fingerprint.items():
        # Check if this hash exists is in hash map
        if hash_value in hash_map:
            
            # Get dictionary of track IDs and their time offsets associated with this hash
            track_entries = hash_map[hash_value]
            
            # Convert query times to numpy array
            query_times_array = np.array(query_times)
            
            # Process each track's time offsets
            for track_id, db_times in track_entries.items():
                db_times_array = np.array(db_times)
                
                # Vectorised calculation of all offsets between this track and query
                # Broadcasting: each db_time - each query_time
                offsets = db_times_array.reshape(-1, 1) - query_times_array.reshape(1, -1)
                
                # Flatten and add all offsets to the matches for this track
                flat_offsets = offsets.flatten()
                matches[track_id].extend(flat_offsets)
    
    # Analyse matches to find the most likely track
    results = []
    
    for track_id, offsets in matches.items():
        # Count frequencies of offsets
        offset_counts = Counter(offsets)
        if not offset_counts:
            continue
        
        # Find most common offset
        best_offset, best_score = offset_counts.most_common(1)[0]
        
        # Convert offset to seconds
        offset_seconds = best_offset * args.hop_length / args.sample_rate
        
        # Calculate confidence score
        if len(offsets) <= 1:
            confidence_score = 0.0
        else:
            confidence_score = (best_score-1) / (len(offsets)-1)
        
        # Add to results with confidence score
        results.append((track_id, best_score, offset_seconds, confidence_score))
    
    # Sort by appropriate metric
    if args.rank_by == 'confidence':
        results.sort(key=lambda x: x[3], reverse=True)  # Sort by confidence
    else:
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    
    return results

def get_all_wav_files(directory):
    """
    Get all .wav files from a directory.
    
    Args:
        directory: Path to the directory containing .wav files
    
    Returns:
        List of paths to all .wac files in the directory
    """
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith('.wav')
    ]


def fingerprintBuilder(database_path, fingerprints_path=None, filenames=None, args=None):
    """
    Create fingerprints from a database of audio files and optionally save them.
    
    Args:
        database_path: Path to the database directory
        fingerprints_path: Path to save fingerprints file (if None, won't save to disk)
        filenames: List of specific filenames to process (default: all files in database_path)
        args: The command-line arguments containing configuration (optional)
    
    Returns:
        Tuple of (hash_map, fingerprints_path) - hash_map is the fingerprint database,
        fingerprints_path is the path where it was saved (or None if not saved)
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Hash Fingerprint Builder")
        config = yaml_config_hook("./config/config.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
        args = parser.parse_args([])
    
    # Get database files
    if filenames is None:
        database_files = get_all_wav_files(database_path)
    else:
        database_files = [os.path.join(database_path, f) for f in filenames]
    
    print(f"Building fingerprints for {len(database_files)} files from {database_path}")
    
    # Build the hash map with args
    hash_map = build_database_hash_map(database_files, args)
    
    # Save the hash map if a path is provided
    if fingerprints_path is not None:
        # Ensure directory exists for fingerprints file
        os.makedirs(os.path.dirname(fingerprints_path), exist_ok=True)
        
        # Convert defaultdict with lambda to regular dict for pickling
        # Apparently pickle can't serialise lambda functions
        pickle_hash_map = {}
        for hash_val, track_dict in hash_map.items():
            pickle_hash_map[hash_val] = dict(track_dict)
        
        # Save the hash map to the specified file path
        # Use gzip compression if filename ends with .gz, otherwise use protocol 4 for speed
        if fingerprints_path.endswith('.gz'):
            with gzip.open(fingerprints_path, 'wb') as f:
                pickle.dump(pickle_hash_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Fingerprints saved to {fingerprints_path} (compressed)")
        else:
            with open(fingerprints_path, 'wb') as f:
                pickle.dump(pickle_hash_map, f, protocol=4)  # Protocol 4 is faster for large objects
            print(f"Fingerprints saved to {fingerprints_path}")
    else:
        print("Fingerprints built but not saved to disk (no path provided)")
    
    return hash_map, fingerprints_path

def audioIdentification(query_path, fingerprints_path=None, output_path="results/matches.txt", query_files=None, args=None, hash_map=None):
    """
    Identify query audio files using stored fingerprints and write results to file.
    
    Args:
        query_path: Path to the query directory
        fingerprints_path: Path to the stored fingerprints file (if None, hash_map must be provided)
        output_path: Path to the output results file
        query_files: List of specific query files to process (default: all files in query_path)
        args: Command line arguments with configuration (optional)
        hash_map: Pre-computed hash map (used if fingerprints_path is None)
    
    Returns:
        Path to the output file
    """
    # Load config
    if args is None:
        parser = argparse.ArgumentParser(description="Hash Audio Identification")
        config = yaml_config_hook("./config/config.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
        args = parser.parse_args([])
    
    # Load the hash map from provided hash_map or from file
    if hash_map is not None:
        print("Using provided hash map (not loaded from file)")
    elif fingerprints_path is not None:
        # Support both compressed and uncompressed files
        if fingerprints_path.endswith('.gz'):
            with gzip.open(fingerprints_path, 'rb') as f:
                hash_map = pickle.load(f)
        else:
            with open(fingerprints_path, 'rb') as f:
                hash_map = pickle.load(f)
        print(f"Loaded fingerprints from {fingerprints_path}")
    else:
        raise ValueError("Either hash_map or fingerprints_path must be provided")
    
    # Get query files
    if query_files is None:
        query_files = get_all_wav_files(query_path)
    else:
        query_files = [os.path.join(query_path, f) for f in query_files]
    
    print(f"Identifying {len(query_files)} query files using fingerprints from {fingerprints_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process each query and write results
    with open(output_path, 'w') as out_file:
        for query_idx, query_file in enumerate(query_files):
            print(f"Processing query file #{query_idx}/{len(query_files)}: {os.path.basename(query_file)}...")
            # Get matches for this query
            identification_results = identify_query(query_file, hash_map, args, plot=False)
            
            # Get top 3 matches
            top_matches = identification_results[:min(3, len(identification_results))]
            
            # Get just the track IDs
            top_track_ids = [track_id for track_id, _, _, _ in top_matches]
            
            # Write query and matches to output file in tab-separated format
            query_basename = os.path.basename(query_file)
            matches_str = '\t'.join(top_track_ids)
            out_file.write(f"{query_basename}\t{matches_str}\n")
    
    print(f"Identification results written to {output_path}")
    return output_path