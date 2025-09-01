from identify import fingerprintBuilder, audioIdentification
from time import time

start_time = time()


hash_map, _ = fingerprintBuilder(
    database_path="database_recordings/",
    fingerprints_path=None  # Don't save to disk
)

fingerprint_done_time = time()
fingerprint_duration = fingerprint_done_time - start_time
print(f"Fingerprinting duration: {fingerprint_duration:.2f} seconds")

output_path = "results/matches.txt"

audioIdentification(
    query_path="query_recordings/",
    output_path=output_path,
    hash_map=hash_map
)

end_time = time()

print(f"Fingerprinting duration: {fingerprint_duration:.2f} seconds")
print(f"Audio identification duration: {end_time - fingerprint_done_time:.2f} seconds")
print(f"Total duration: {end_time - start_time:.2f} seconds")

# Analyse results
with open(output_path, 'r') as f:
    lines = f.readlines()

total = len(lines)
top1_correct = 0
top3_correct = 0

for line in lines:
    parts = line.strip().split('\t')
    query = parts[0]
    matches = parts[1:] if len(parts) > 1 else []
    true_track = query.split('-snippet')[0] + '.wav'
    
    if true_track in matches:
        position = matches.index(true_track) + 1
        if position == 1:
            top1_correct += 1
        if position <= 3:
            top3_correct += 1

print(f"\nAccuracy:")
print(f"Top-1: {top1_correct}/{total} ({top1_correct/total*100:.1f}%)")
print(f"Top-3: {top3_correct}/{total} ({top3_correct/total*100:.1f}%)")