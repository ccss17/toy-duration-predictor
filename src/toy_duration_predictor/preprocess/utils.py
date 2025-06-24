from pathlib import Path

import pandas as pd
import midii


def get_files(dir_path, type, sort=False):
    paths = Path(dir_path).rglob(f"*.{type}")
    if sort:
        return sorted(paths, key=lambda p: p.stem)
    else:
        return paths


def _preprocess_remove_front_back_silence(df):
    is_valid_lyric = df["lyric"] != " "
    valid_indices = df.index[is_valid_lyric].tolist()
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    df = df.iloc[first_valid_idx : last_valid_idx + 1].reset_index(drop=True)
    return df


def _preprocess_sort_by_start_time(df):
    df = df.sort_values(by="start_time").reset_index(drop=True)
    return df


def _preprocess_remove_front_back_silence(df):
    is_valid_lyric = df["lyric"] != " "
    valid_indices = df.index[is_valid_lyric].tolist()
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    df = df.iloc[first_valid_idx : last_valid_idx + 1].reset_index(drop=True)
    return df


def _preprocess_silence_pitch_zero(df):
    df.loc[df["lyric"] == " ", "pitch"] = 0
    return df


def _preprocess_merge_silence(df):
    output_notes = []
    i = 0
    n = len(df)
    while i < n:
        current_row = df.iloc[i]  # Pandas Series
        if current_row["lyric"] == " ":
            merged_start_time = current_row["start_time"]
            merged_end_time = current_row["end_time"]

            j = i + 1
            while j < n and df.iloc[j]["lyric"] == " ":
                merged_end_time = df.iloc[j][
                    "end_time"
                ]  # 마지막 공백의 end_time으로 업데이트
                j += 1

            merged_item = {
                "start_time": merged_start_time,
                "end_time": merged_end_time,
                "pitch": 0,
                "lyric": " ",
                "duration": merged_end_time - merged_start_time,
            }
            output_notes.append(merged_item)
            i = j  # 병합된 블록 다음으로 인덱스 이동
        else:
            non_space_item = {
                "start_time": current_row["start_time"],
                "end_time": current_row["end_time"],
                "pitch": current_row["pitch"],
                "lyric": current_row["lyric"],
                "duration": current_row["duration"],
            }
            output_notes.append(non_space_item)
            i += 1
    df = pd.DataFrame(output_notes)
    return df


def _preprocess_remove_short_silence(df, threshold=0.3):
    processed_notes = []
    absorbed_time = 0.0

    for i in range(len(df)):
        current_note_s = df.iloc[i]
        if (
            current_note_s["lyric"] == " "
            and current_note_s["duration"] < threshold
        ):
            absorbed_time += current_note_s["duration"]
            continue
        else:
            note_to_add = current_note_s.to_dict()
            if absorbed_time > 0:
                note_to_add["start_time"] -= absorbed_time
                note_to_add["duration"] = (
                    note_to_add["end_time"] - note_to_add["start_time"]
                )
                absorbed_time = 0.0
            processed_notes.append(note_to_add)

    df = pd.DataFrame(processed_notes)
    return df


def _preprocess_add_quantized_duration_col(df, ticks_per_beat, unit="32"):
    unit_tick = midii.beat2tick(
        midii.NOTE[f"n/{unit}"].beat, ticks_per_beat=ticks_per_beat
    )
    df["quantized_duration"], _ = midii.quantize(
        df["duration"].values, unit=unit_tick
    )

    return df
