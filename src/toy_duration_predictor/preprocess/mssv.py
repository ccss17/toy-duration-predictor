from pathlib import Path
import re
from typing import List, Dict, Any

import ray
import pandas as pd
import midii

from .utils import (
    _preprocess_sort_by_start_time,
    _preprocess_remove_front_back_silence,
    _preprocess_silence_pitch_zero,
    _preprocess_merge_silence,
    _preprocess_remove_short_silence,
    _preprocess_add_quantized_duration_col,
)


def singer_id_from_filepath(filepath):
    return int(re.findall(r"s\d\d", filepath)[0][1:])


def midi_to_note_list(midi_filepath, quantize=False):
    try:
        mid = midii.MidiFile(
            midi_filepath, convert_1_to_0=True, lyric_encoding="utf-8"
        )
        mid.lyrics
    except:  # noqa: E722
        mid = midii.MidiFile(
            midi_filepath, convert_1_to_0=True, lyric_encoding="cp949"
        )
    if quantize:
        mid.quantize(unit="32")
    data = []
    total_duration = 0
    residual_duration = 0
    active_note = {}
    silence_note = {}
    for msg in mid.tracks[0]:
        msg_end_time = total_duration + msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            residual_duration += msg.time
            if residual_duration > 0:
                if not silence_note:
                    silence_note = {
                        "start_time": total_duration,
                        "pitch": 0,
                        "lyric": " ",
                    }
                silence_note["end_time"] = msg_end_time
                silence_note["duration"] = (
                    msg_end_time - silence_note["start_time"]
                )
                data.append(silence_note.copy())
                silence_note.clear()
                residual_duration = 0
            active_note = {
                "start_time": msg_end_time,
                "pitch": msg.note,
            }
        elif msg.type == "lyrics":
            active_note["lyric"] = midii.MessageAnalyzer_lyrics(
                msg=msg, encoding=mid.lyric_encoding
            ).lyric
        elif msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        ):
            active_note["end_time"] = msg_end_time
            active_note["duration"] = msg_end_time - active_note["start_time"]
            data.append(active_note.copy())
            active_note.clear()
        else:
            if not active_note and not silence_note:
                silence_note = {
                    "start_time": total_duration,
                    "pitch": 0,
                    "lyric": " ",
                }
            if not active_note:
                residual_duration += msg.time
        total_duration = msg_end_time

    return data, mid.ticks_per_beat


def _preprocess_slice_actual_lyric(df):
    j_indices = df.index[df["lyric"] == "J"].tolist()
    idx_j = j_indices[0]
    h_indices = df.index[df["lyric"] == "H"].tolist()
    idx_h = h_indices[0]
    slice_start_index = idx_j + 1
    slice_end_index = idx_h
    df = df.iloc[slice_start_index:slice_end_index].reset_index(drop=True)
    return df


def preprocess_notes(notes, ticks_per_beat, unit="32"):
    df = pd.DataFrame(notes)

    # ["J":"H"]
    df = _preprocess_slice_actual_lyric(df)
    # sort by time
    df = _preprocess_sort_by_start_time(df)
    # remove front & back silence
    df = _preprocess_remove_front_back_silence(df)
    # lyric=" " --> pitch=0
    df = _preprocess_silence_pitch_zero(df)
    # merge lyric=" " items
    df = _preprocess_merge_silence(df)
    # remove silence < 0.3
    df = _preprocess_remove_short_silence(df, 0.3)
    #
    df = _preprocess_add_quantized_duration_col(df, ticks_per_beat, unit=unit)
    return df


def process_midi_flat_map(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes a single file path. Returns a list containing one dictionary on success,
    or an empty list on failure. Designed for use with flat_map().
    """
    file_path_str = row["path"]
    try:
        mssv_sample_list, ticks_per_beat = midi_to_note_list(file_path_str)
        df = preprocess_notes(mssv_sample_list, ticks_per_beat=ticks_per_beat)
        singer_id = singer_id_from_filepath(file_path_str)

        durations = df["duration"].tolist()
        quantized_durations = df["quantized_duration"].tolist()

        return [
            {
                "durations": durations,
                "quantized_durations": quantized_durations,
                "singer_id": singer_id,
            }
        ]
    except Exception as e:
        print(f"CRITICAL ERROR processing {file_path_str}: {e}")
        return []


def preprocess_dataset(midi_file_directory, output_parquet_path):
    context = ray.init()
    print(context.dashboard_url)
    print(f"Ray cluster started: {ray.cluster_resources()}")

    all_midi_paths = list(Path(midi_file_directory).rglob("*.mid"))
    print(f"Found {len(all_midi_paths)} MIDI files to process.")

    # --- Step 2: Create a dataset of file paths ---
    # This is the idiomatic way. We create a dataset where each row is a path.
    # We pass a list of dictionaries to give the column a name: "path".
    print("Creating initial dataset of file paths...")
    ds = ray.data.from_items([{"path": str(p)} for p in all_midi_paths])

    # --- Step 3: Use .map() to process files in parallel ---
    # .map() applies the function to each row of the dataset in parallel.
    # Ray Data manages the tasks, memory, and scheduling for you.
    print("Applying parallel processing function using .map()...")
    processed_ds = ds.flat_map(process_midi_flat_map)

    # --- Step 4: Filter out any rows that failed ---
    # Our function returns an empty dict on failure, so we filter those out.
    # processed_ds = processed_ds.filter(lambda row: row)

    # Now, let's inspect the schema of the PROCESSED dataset
    print("\nProcessed Dataset schema:")
    print(processed_ds.schema())  # Corrected print statement

    print("\nFirst 1 rows of processed data:")
    processed_ds.show(1)

    # You chose to repartition to 1, which is fine if the total dataset is < 1GB.
    # This will create a single Parquet file in the output directory.
    print("Repartitioning dataset...")
    processed_ds = processed_ds.repartition(num_blocks=1)

    print(f"\nWriting dataset to Parquet format at: {output_parquet_path}")
    processed_ds.write_parquet(output_parquet_path)

    # This will now show the correct number of processed files (rows in the dataset)
    print(
        f"\nProcessing complete! {processed_ds.count()} files successfully processed."
    )

    ray.shutdown()
