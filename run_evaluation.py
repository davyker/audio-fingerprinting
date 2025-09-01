from identify import *
import re
import sys
import yaml
import time
import random
import pandas as pd
from visualisation import plot_evaluation_results

def run_evaluation(args, num_queries=10, num_distractors=None, plot=False):
    """
    Run an evaluation of the fingerprinting system.
    
    Args:
        args: Command line arguments with configuration
        num_queries: Number of query files to test, or "all" for all query files
        num_distractors: Number of distractor files to include, or "all" for all available files,
                         or None to only include files that have a corresponding query
        plot: Whether to show plots
        
    Returns:
        DataFrame with evaluation results
    """
    
    run_evaluation.run_dir = None
    # Get all query files
    all_query_files = get_all_wav_files(args.query_dir)
    
    # Handle case for all queries
    if num_queries == "all":
        test_query_files = all_query_files
    elif isinstance(num_queries, int):
        # Ensure num_queries is not greater than available files
        num_queries = min(num_queries, len(all_query_files))
        test_query_files = random.sample(all_query_files, num_queries)
    else:
        raise ValueError("num_queries must be 'all' or an integer")
    
    print(f"Selected {len(test_query_files)} query files for testing")
    
    # Find the corresponding database files for selected queries
    selected_db_files = []
    valid_query_files = []
    
    print("Matching query files with database files...")
    for query_file in test_query_files:
        db_file = get_matching_database_file(query_file, args.database_dir)
        if db_file is not None:
            if db_file not in selected_db_files:
                selected_db_files.append(db_file)
            if query_file not in valid_query_files:
                valid_query_files.append(query_file)
        else:
            print(f"Warning: No matching database file found for {query_file}, skipping...")
    
    # Add distractor files
    all_database_files = get_all_wav_files(args.database_dir)
    distractor_candidates = [f for f in all_database_files if f not in selected_db_files]
    
    if num_distractors == "all":
        # Use all available distractors
        distractor_files = distractor_candidates
    elif isinstance(num_distractors, int):
        # Use a random subset of distractors
        num_distractors = min(num_distractors, len(distractor_candidates))
        distractor_files = random.sample(distractor_candidates, num_distractors)
    else:
        print(f"num_distractors not specified, only using files that have a corresponding query")
        distractor_files = []
    
    # Combine selected db files and distractors
    all_test_db_files = selected_db_files + distractor_files
    
    print(f"Building hash map for {len(all_test_db_files)} database files:")
    print(f"- {len(selected_db_files)} files corresponding to test queries")
    print(f"- {len(distractor_files)} distractor files")
    
    # Build the combined hash map
    hash_map = build_database_hash_map(all_test_db_files, args)
    
    # Evaluate each query
    print("\nRunning identification on queries...")
    results = []
    correct_count = 0
    total_count = 0
        
    for query_idx, query_file in enumerate(valid_query_files):
        true_db_file = get_matching_database_file(query_file, args.database_dir)
        true_track_id = os.path.basename(true_db_file)
        
        print(f"\nQuery {query_idx+1}/{len(valid_query_files)}: {os.path.basename(query_file)}", end=' ... ')
        
        # Identify the track
        identification_results = identify_query(query_file, hash_map, args, plot=plot)
        
        # Check if the true match is in the results and get its position
        true_match_position = None
        true_match_confidence_position = None
        
        for i, (track_id, score, offset, confidence) in enumerate(identification_results):
            if track_id == true_track_id:
                true_match_position = i + 1
                break
        
        # Also check where the true match would rank if sorted by confidence
        confidence_sorted = sorted(identification_results, key=lambda x: x[3], reverse=True)
        for i, (track_id, score, offset, confidence) in enumerate(confidence_sorted):
            if track_id == true_track_id:
                true_match_confidence_position = i + 1
                break
                
        # Record results
        correct = (true_match_position == 1)
        correct_by_confidence = (true_match_confidence_position == 1)
        correct_count += int(correct)
        total_count += 1
        
        if true_match_position is None:
            print(f"There were no hash matches with {true_track_id}!")
        
        # Extract genre from query filename
        genre = os.path.basename(query_file).split('.')[0]  # 'pop' or 'classical'
        
        result = {
            'query_file': os.path.basename(query_file),
            'true_match': true_track_id,
            'identification_results': identification_results,
            'identified_as': identification_results[0][0] if identification_results else None,
            'true_match_position': true_match_position,
            'true_match_confidence_position': true_match_confidence_position,
            'correct': correct,
            'correct_by_confidence': correct_by_confidence,
            'top_score': identification_results[0][1] if identification_results else 0,
            'top_confidence': identification_results[0][3] if identification_results else 0,
            'true_match_score': next((score for track, score, _, _ in identification_results 
                                     if track == true_track_id), 0),
            'true_match_confidence': next((confidence for track, _, _, confidence in identification_results 
                                         if track == true_track_id), 0),
            'genre': genre
        }
        results.append(result)
        
        # Print top results with scores and confidence
        top_n = 3
        print(f"Top {top_n} matches:")
        for i, (track, score, offset, confidence) in enumerate(identification_results[:top_n]):
            is_true_match = " (TRUE MATCH)" if track == true_track_id else ""
            print(f"  {i+1}. {track} - Score: {score}, Confidence: {confidence:.4f}, Offset: {offset:.2f} seconds{is_true_match}")
            
        
        # If true match is not in top_n or not found at all
        if true_match_position is None:
            print(f"The query had 0 matches with the true track: {true_track_id}")
        elif true_match_position > top_n:
            # True match found but not in top_n
            # Find its details
            position_idx = true_match_position - 1  # Convert to 0-based index
            if position_idx < len(identification_results):
                track, score, offset, confidence = identification_results[position_idx]
                print(f"  ...")
                print(f"  {true_match_position}. {track} - Score: {score}, Confidence: {confidence:.4f}, Offset: {offset:.2f} seconds (TRUE MATCH)")
        
        # Check if confidence-based ranking would have resulted in a correct match
        if not correct and true_match_confidence_position == 1:
            print("MATCH WOULD HAVE BEEN CORRECT IF CONFIDENCE USED")
            
        print(f"{'      TRUE' if correct else '      FALSE - '} ... True match position: --{true_match_position}--")

    # Output redirection class to capture summary to file
    class TeeOutput:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.stdout
            
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
            
        def flush(self):
            self.file.flush()
            self.stdout.flush()
            
        def __getattr__(self, attr):
            # Pass any other attributes to self.stdout
            return getattr(self.stdout, attr)
    
    # Create run directory
    if run_evaluation.run_dir is None:
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Find the highest run number
        run_dirs = [d for d in os.listdir("results") if re.match(r"run\d+", d)]
        if run_dirs:
            max_run = max([int(re.findall(r"\d+", d)[0]) for d in run_dirs])
            new_run = max_run + 1
        else:
            new_run = 0
        
        # Create new run directory
        run_evaluation.run_dir = os.path.join("results", f"run{new_run}")
        os.makedirs(run_evaluation.run_dir, exist_ok=True)
        
        # Save config.yaml for this run
        config_path = os.path.join(run_evaluation.run_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    
    # Redirect stdout to also write to a file
    summary_path = os.path.join(run_evaluation.run_dir, "evaluation_summary.txt")
    summary_file = open(summary_path, 'w')
    sys.stdout = TeeOutput(summary_file)
    
    # Calculate all metrics from the results list
    
    # Top-1 accuracy
    total_count = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Top-3 accuracy
    top3_correct_count = sum(1 for r in results if r['true_match_position'] is not None and r['true_match_position'] <= 3)
    top3_accuracy = top3_correct_count / total_count if total_count > 0 else 0
    
    # Rank metrics
    ranks = [r['true_match_position'] for r in results if r['true_match_position'] is not None]
    mean_rank = sum(ranks) / len(ranks) if ranks else 0
    median_rank = np.median(ranks) if ranks else 0
    
    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = [1.0 / r['true_match_position'] if r['true_match_position'] is not None else 0.0 for r in results]
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
    # Genre-specific metrics
    pop_results = [r for r in results if r['genre'] == 'pop']
    classical_results = [r for r in results if r['genre'] == 'classical']
    
    pop_total = len(pop_results)
    pop_correct = sum(1 for r in pop_results if r['correct'])
    pop_accuracy = pop_correct / pop_total if pop_total > 0 else 0
    
    classical_total = len(classical_results)
    classical_correct = sum(1 for r in classical_results if r['correct'])
    classical_accuracy = classical_correct / classical_total if classical_total > 0 else 0
    
    # Genre-specific MRR
    pop_rr = [1.0 / r['true_match_position'] if r['true_match_position'] is not None else 0.0 for r in pop_results]
    pop_mrr = sum(pop_rr) / len(pop_rr) if pop_rr else 0
    
    classical_rr = [1.0 / r['true_match_position'] if r['true_match_position'] is not None else 0.0 for r in classical_results]
    classical_mrr = sum(classical_rr) / len(classical_rr) if classical_rr else 0
    
    print(f"\nEvaluation Summary:")
    print(f"Overall Accuracy (Top-1): {accuracy:.2f} ({correct_count}/{total_count})")
    print(f"Overall Accuracy (Top-3): {top3_accuracy:.2f} ({top3_correct_count}/{total_count})")
    print(f"Pop Accuracy: {pop_accuracy:.2f} ({pop_correct}/{pop_total})")
    print(f"Classical Accuracy: {classical_accuracy:.2f} ({classical_correct}/{classical_total})")
    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"Median Rank: {median_rank:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank:.4f} {f'(Harmonic Mean Rank: {1.0/mean_reciprocal_rank:.2f})' if mean_reciprocal_rank > 0 else ''}")
    
    if pop_mrr > 0:
        print(f"Pop MRR: {pop_mrr:.4f} (Harmonic Mean Rank: {1.0/pop_mrr:.2f})")
    else:
        print(f"Pop MRR: N/A (no pop samples)")
        
    if classical_mrr > 0:
        print(f"Classical MRR: {classical_mrr:.4f} (Harmonic Mean Rank: {1.0/classical_mrr:.2f})")
    else:
        print(f"Classical MRR: N/A (no classical samples)")
    print(f"Model failed on the following queries:\n{[result['query_file'] for result in results if not result['correct']]}")
    # Flag as potential duplicates all cases where true match was in the top 2
    # and the ratio between the top score and the true match score was between 0.5 and 2
    # and the ratio between the second place score and the true match score was >= 4
    print("Potential duplicates (top 3 matches with score ratio between 0.5 and 2):")
    for result in results:
        if result['true_match_position'] is not None and result['true_match_position'] <= 2:
            top_score = result['top_score']
            second_place_score = result['identification_results'][1][1] if len(result['identification_results']) > 1 else None
            third_place_score = result['identification_results'][2][1] if len(result['identification_results']) > 2 else None
            if None not in [top_score, second_place_score, third_place_score] and top_score / second_place_score <= 2 and second_place_score / third_place_score >= 4:
                print(f"  {result['query_file']}: Top Score: {top_score}, 2nd place score: {second_place_score}", end=' --- ')
                print(f"Correct: {'Yes - 2nd place was '+result['identification_results'][1][0] if result['correct'] else 'No - identified as '+result['identified_as']}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df['genre'] = df['query_file'].apply(lambda x: 'pop' if x.startswith('pop') else 'classical')
    
    # Add overall and genre-specific statistics as DataFrame attributes
    df.attrs['overall_accuracy'] = accuracy
    df.attrs['top3_accuracy'] = top3_accuracy
    df.attrs['pop_accuracy'] = pop_accuracy
    df.attrs['classical_accuracy'] = classical_accuracy
    df.attrs['pop_correct'] = pop_correct
    df.attrs['pop_total'] = pop_total
    df.attrs['classical_correct'] = classical_correct
    df.attrs['classical_total'] = classical_total
    df.attrs['mean_rank'] = mean_rank
    df.attrs['median_rank'] = median_rank
    
    
    return df

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Hashy")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("Running hash map evaluation...")
    
    # Handle special values
    num_queries = args.num_queries
    if isinstance(num_queries, str) and num_queries != "all":
        num_queries = int(num_queries)
        
    num_distractors = args.num_distractors
    if isinstance(num_distractors, str):
        if num_distractors == "all":
            pass  # Keep as "all"
        elif num_distractors == "0":
            num_distractors = 0
        else:
            num_distractors = int(num_distractors)
    
    # Run evaluation - this will create the run directory only when needed
    df = run_evaluation(args, num_queries=num_queries, num_distractors=num_distractors, plot=False)
    
    # Check if run_dir was created (if evaluation was completed)
    if hasattr(run_evaluation, 'run_dir') and run_evaluation.run_dir is not None:
        run_dir = run_evaluation.run_dir
        
        # Save results to CSV in the run directory
        csv_path = os.path.join(run_dir, "hash_map_evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to {csv_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Complete. Elapsed time: {elapsed_time:.2f} seconds - {elapsed_time/60:.2f} minutes")

        # Plot evaluation results and save to run directory
        plot_path = os.path.join(run_dir, "evaluation_plot.png")
        plot_evaluation_results(df, save_path=plot_path)
        
        # Restore stdout to its original state
        original_stdout = sys.stdout.stdout
        summary_file = sys.stdout.file
        sys.stdout = original_stdout
        summary_file.close()
        
        print(f"Run results saved in directory: {run_dir}")
        
        # Reset the run_dir for next run
        run_evaluation.run_dir = None
    else:
        if hasattr(sys.stdout, 'stdout'):
            original_stdout = sys.stdout.stdout
            sys.stdout = original_stdout