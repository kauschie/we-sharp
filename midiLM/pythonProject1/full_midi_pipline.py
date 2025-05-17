import os
import csv
from collections import defaultdict
import concurrent.futures
import threading
from tqdm import tqdm

# =================== Global Configuration ===================
INPUT_FOLDER_RAW = r"../MIDI_TXT"
FOLDER_CLEANED = r"../cleaned_txt"
FOLDER_MAPPING = r"../mapping_csv"
FOLDER_MELODY_ONLY = r"../melody_only_output"
LOG_NOTEON_DEDUP = "outputlog/noteon_dedup_log.txt"
LOG_CHORD_DELETION = "deletion_log.txt"
CHORD_STATS_FILE = "chord_stats.csv"
START_INDEX = "000001"  # Only process files >= this index

# Ensure output folders exist
for folder in [FOLDER_CLEANED, FOLDER_MAPPING, FOLDER_MELODY_ONLY, "outputlog"]:
    os.makedirs(folder, exist_ok=True)

# =============== Step 1: Sort zero-offset note blocks ===============
def sort_and_write_block(block, first_offset, outfile):
    block.sort(key=lambda x: x[1])  # Sort by note pitch
    block[0] = (block[0][0], block[0][1], block[0][2], first_offset)
    for event in block:
        outfile.write(f"{event[0]} {event[1]} {event[2]} {event[3]}\n")

def process_midi_text_file(input_txt, output_txt):
    with open(input_txt, "r") as infile, open(output_txt, "w") as outfile:
        lines = infile.readlines()
        block, flag, first_offset = [], 0, 0

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            event_type, note, velocity, offset = parts[0], int(parts[2]), int(parts[3]), int(parts[4])

            next_offset = int(lines[idx+1].strip().split()[4]) if idx+1 < len(lines) and len(lines[idx+1].strip().split()) >= 5 else -1

            if flag == 0:
                if next_offset == 0:
                    flag = 1
                    first_offset = offset
                    block.append((event_type, note, velocity, 0))
                else:
                    outfile.write(f"{event_type} {note} {velocity} {offset}\n")
            else:
                block.append((event_type, note, velocity, offset))
                if next_offset != 0:
                    sort_and_write_block(block, first_offset, outfile)
                    block, flag = [], 0

        if block:
            sort_and_write_block(block, first_offset, outfile)

def process_all_midi_files(input_folder, output_folder):
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    print(f"🎵 Found {len(txt_files)} MIDI TXT files. Sorting...")
    for filename in txt_files:
        process_midi_text_file(os.path.join(input_folder, filename), os.path.join(output_folder, filename))
    print("✅ Sorting completed.\n")

# =============== Step 2: Create note_on/off mapping CSV ===============
def create_mapping_csv(input_txt, output_csv):
    visit_list = set()
    with open(input_txt, "r") as infile, open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["note", "on_line", "off_line", "duration"])
        lines = infile.readlines()

        for line_num, line in enumerate(lines):
            if line_num in visit_list: continue
            parts = line.strip().split()
            if len(parts) < 4: continue

            event_type, note, offset = parts[0], int(parts[1]), int(parts[3])
            if event_type != "note_on": continue

            on_line, temp_offset = line_num + 1, 0
            for search_num in range(line_num + 1, len(lines)):
                search_parts = lines[search_num].strip().split()
                if len(search_parts) < 4: continue
                search_event_type, search_note, search_offset = search_parts[0], int(search_parts[1]), int(search_parts[3])
                temp_offset += search_offset

                if search_event_type == "note_off" and search_note == note:
                    writer.writerow([note, on_line, search_num + 1, temp_offset])
                    visit_list.update({line_num, search_num})
                    break

def process_all_txt_files(input_folder, output_folder):
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    print(f"🧠 Mapping {len(txt_files)} files...")
    for filename in txt_files:
        create_mapping_csv(os.path.join(input_folder, filename), os.path.join(output_folder, filename.replace(".txt", ".csv")))
    print("✅ Mapping completed.\n")

# =============== Step 3: Deduplicate broken note_on events ===============
def deduplicate_note_on_inplace(input_txt, log_entries):
    with open(input_txt, "r") as infile:
        lines = infile.readlines()

    to_delete = set()
    file_name, total_lines, i = os.path.basename(input_txt), len(lines), 0

    while i < total_lines:
        parts = lines[i].strip().split()
        if len(parts) < 4 or parts[0] != "note_on":
            i += 1
            continue

        note = int(parts[1])
        found_on2, on2_line, off1_line = False, -1, -1

        for j in range(i + 1, total_lines):
            next_parts = lines[j].strip().split()
            if len(next_parts) < 4 or int(next_parts[1]) != note:
                continue
            if next_parts[0] == "note_on" and not found_on2:
                found_on2, on2_line = True, j
            elif next_parts[0] == "note_off":
                off1_line = j
                break

        if found_on2 and off1_line != -1:
            to_delete.update({on2_line, off1_line})
            log_entries.extend([
                f"[{file_name}] DELETE on2 at line {on2_line+1} (note {note})",
                f"[{file_name}] DELETE off1 at line {off1_line+1} (note {note})"
            ])
        i += 1

    for idx in sorted(to_delete):
        if len(lines[idx].strip().split()) >= 4:
            offset_to_transfer = int(lines[idx].strip().split()[3])
            next_idx = idx + 1
            if next_idx < len(lines) and next_idx not in to_delete:
                parts = lines[next_idx].strip().split()
                if len(parts) >= 4:
                    parts[3] = str(int(parts[3]) + offset_to_transfer)
                    lines[next_idx] = " ".join(parts) + "\n"

    with open(input_txt, "w") as outfile:
        for idx, line in enumerate(lines):
            if idx not in to_delete:
                outfile.write(line)

    return len(to_delete)

def process_all_files(input_folder, log_path):
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    all_logs = []

    for filename in files:
        full_path = os.path.join(input_folder, filename)
        round_count = 0
        while True:
            round_count += 1
            if deduplicate_note_on_inplace(full_path, all_logs) == 0:
                break
        print(f"✅ {filename} cleaned in {round_count} rounds")

    with open(log_path, "w", encoding="utf-8") as log_file:
        for entry in all_logs:
            log_file.write(entry + "\n")
    print("✅ note_on de-duplication completed.\n")

# =============== Step 4: Remove chords and write melody-only output ===============
log_lock = threading.Lock()
stat_lock = threading.Lock()
chord_counter = defaultdict(int)
all_logs = []

def record_log(message):
    with log_lock:
        all_logs.append(message)

def update_chord_stats(chord_notes):
    chord_key = "_".join(map(str, sorted(chord_notes)))
    with stat_lock:
        chord_counter[chord_key] += 1

def process_file(file):
    txt_path = os.path.join(FOLDER_CLEANED, file)
    csv_path = os.path.join(FOLDER_MAPPING, file.replace(".txt", ".csv"))
    output_path = os.path.join(FOLDER_MELODY_ONLY, file)

    with open(txt_path, "r") as f:
        lines = f.readlines()
    with open(csv_path, "r") as csvfile:
        mapping = {int(row["on_line"]) - 1: (int(row["note"]), int(row["off_line"]) - 1, int(row["duration"]))
                   for row in csv.DictReader(csvfile)}

    to_delete = set()
    idx = 0
    while idx < len(lines):
        block = []
        offset = int(lines[idx].strip().split()[3]) if len(lines[idx].strip().split()) >= 4 else -1
        if offset != 0:
            idx += 1
            continue
        block.append(idx)
        j = idx + 1
        while j < len(lines) and int(lines[j].strip().split()[3]) == 0:
            block.append(j)
            j += 1

        durations = defaultdict(list)
        for line_idx in block:
            if line_idx in mapping:
                note, off_line, dur = mapping[line_idx]
                durations[dur].append((note, line_idx, off_line))

        for dur, group in durations.items():
            if len(group) >= 3:
                notes = [x[0] for x in group]
                update_chord_stats(notes)
                for _, on, off in group:
                    to_delete.update([on, off])
                    record_log(f"[{file}] DELETE on_line {on+1}, off_line {off+1}, note {mapping[on][0]}")

        idx = j if j > idx else idx + 1

    with open(output_path, "w") as out_f:
        for idx, line in enumerate(lines):
            if idx in to_delete:
                offset = int(line.strip().split()[3]) if len(line.strip().split()) >= 4 else 0
                next_idx = idx + 1
                if next_idx < len(lines) and next_idx not in to_delete:
                    parts = lines[next_idx].strip().split()
                    if len(parts) >= 4:
                        parts[3] = str(int(parts[3]) + offset)
                        lines[next_idx] = " ".join(parts) + "\n"
                continue
            out_f.write(line)

def flush_chord_stats():
    with open(CHORD_STATS_FILE, "w", encoding="utf-8") as f:
        f.write("Chord,Count\n")
        for chord, count in sorted(chord_counter.items(), key=lambda x: (len(x[0].split("_")), x[0])):
            f.write(f"{chord},{count}\n")

def process_all_parallel():
    files = [f for f in os.listdir(FOLDER_CLEANED) if f.endswith(".txt") and f.split(".")[0] >= START_INDEX]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files), desc="🎶 Melody Cleaning", unit="file"))
    with open(LOG_CHORD_DELETION, "w", encoding="utf-8") as f:
        for entry in all_logs:
            f.write(entry + "\n")
    flush_chord_stats()
    print("✅ Chord removal and melody-only export completed.\n")

# =============== MAIN PIPELINE ===============
def main():
    print("\n🎵 Step 1: Sorting MIDI Events...")
    process_all_midi_files(INPUT_FOLDER_RAW, FOLDER_CLEANED)

    print("🧠 Step 2: Creating note_on/off mappings...")
    process_all_txt_files(FOLDER_CLEANED, FOLDER_MAPPING)

    print("🧹 Step 3: Deduplicating note_on conflicts...")
    process_all_files(FOLDER_CLEANED, LOG_NOTEON_DEDUP)

    print("🎼 Step 4: Removing chords and generating melody-only data...")
    process_all_parallel()

    print("\n🎉 All preprocessing steps completed!")

if __name__ == "__main__":
    main()
