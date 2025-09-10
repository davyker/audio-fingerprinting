# Audio Fingerprinting System

A Python implementation of audio fingerprinting for music identification, using spectral peaks and hash-based matching.
This is based on the method described in "An Industrial-Strength Audio Search Algorithm" (Avery Li-Chun Wang 2003).

## Method

The images below explain the fundamental concept.
Each database track is converted to a spectrogram and peak-picking is performed with a fixed minimum distance between points, as below.

![Peaks Example](https://github.com/user-attachments/assets/82796919-a89e-477c-a7a3-93f1e9300fc9)

Each peak is used as an anchor point with which to create hashes (pairs of peaks). Each other peak within a target zone to the right of the anchor is paired with the anchor to create a hash.

![Hashes Example](https://github.com/user-attachments/assets/9571c755-3db2-47fd-bf40-a05626f46c70)

The idea is that if a query contains a segment of one of the recordings in our database, it will contains pairs of spectral peaks that align with that track far more often than two aritrary recordings, and with a consistent time offset.

## Results

The overall top-1 accuracy is 85.4% (top-3 accuracy 89.2%) when using minimum peak distance of 10, which gives a good tradeoff between speed and accuracy.

![Results](https://github.com/user-attachments/assets/acebc719-a87b-4db6-9f3e-787842446c98)

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
