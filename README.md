# Audio Fingerprinting System

A Python implementation of audio fingerprinting for music identification, using spectral peak analysis and hash-based matching.
This is based on the method described in "An Industrial-Strength Audio Search Algorithm" (Avery Li-Chun Wang 2003).

## Setup

Install dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete fingerprinting and identification pipeline:

```bash
python test.py
```

This will:
1. Build fingerprints from `database_recordings/`
2. Identify queries from `query_recordings/`
3. Save results to `results/matches.txt`
4. Display accuracy metrics

## Configuration

Edit `config/config.yaml` to adjust parameters:
- Peak detection settings (`db_min_distance`, `query_min_distance`)
- Target zone dimensions (`target_zone_width`, `target_zone_height`)
- Hash generation mode (`hash_mode`: 'pairs' or 'points')
- Audio processing parameters (FFT size, hop length, sample rate)

## Evaluation

Run evaluation with test queries:

```bash
python run_evaluation.py --num_queries 10 --num_distractors 50
```

This generates accuracy metrics and plots in the `results/` directory.

## File Structure

- `hash.py` - Core fingerprinting algorithms
- `identify.py` - Database building and query matching
- `run_evaluation.py` - System evaluation framework
- `visualisation.py` - Plotting utilities
- `config/config.yaml` - System parameters