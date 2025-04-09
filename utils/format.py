import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from argparse import ArgumentParser


SKATERS = ['Skater_A', 'Skater_B', 'Skater_C', 'Skater_D']
JUMPS = ['Axel', 'Comb', 'Flip', 'Lutz', 'Salchow', 'Loop', 'Toeloop']

LABEL_MAP = {
        'Lutz': 0,
        'Loop': 1,
        'Axel': 2,
        'Comb': 3,
        'Flip': 4,
        'Salchow': 5,
        'Toeloop': 6
    }

# Save to FACT structure
FEATURES_DIR = Path("./CVPR2024-FACT/data/fsjump/features")
LABELS_DIR = Path("./CVPR2024-FACT/data/fsjump/labels")
SPLITS_DIR = Path("./CVPR2024-FACT/data/fsjump/splits")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def get_marker_data(json_file: str) -> dict:
    with open(json_file, 'r') as f:
        return json.load(f)['Markers']


def get_main_range(parts: List[dict]) -> Tuple[int, int, dict]:
    ranges = [(part['Range']['Start'], part['Range']['End']) for part in parts]
    time_ranges = [end - start for start, end in ranges]

    main_idx = np.argmax(time_ranges)
    main_start, main_end = ranges[main_idx]

    # Adjust to be zero-indexed for array slicing
    return main_start - 1, main_end - 1, parts[main_idx]


def get_time_range(marker_data: List[dict]) -> Tuple[int, int]:
    # Initialize using the first marker's time range
    start, end, _ = get_main_range(marker_data[0]['Parts'])

    for marker in marker_data:
        tmp_start, tmp_end, _ = get_main_range(marker['Parts'])
        start = max(start, tmp_start)
        end = min(end, tmp_end)

    return start, end


def get_pose_array(marker_data: List[dict], time_range: Tuple[int, int]) -> Tuple[np.ndarray, List[str]]:
    start_time, end_time = time_range
    if start_time >= end_time or start_time < 0 or end_time < 0:
        raise ValueError(f"Invalid time range provided: {time_range}")

    labels = []
    pose3d = []

    for marker in marker_data:
        parts = marker['Parts']
        marker_start, _, main_part = get_main_range(parts)

        start = start_time - marker_start
        end = end_time - marker_start

        cood_3d = np.array(main_part['Values'])[start:end, :3]

        labels.append(marker['Name'])
        pose3d.append(cood_3d)

    pose3d = np.array(pose3d).transpose(1, 0, 2)  # (frame, marker, xyz)
    return pose3d, labels


def load_rig_mapping(rig_file: str, rig_name: str, marker_labels: List[str]) -> Tuple[List[str], List[List[int]]]:
    with open(rig_file, "r") as f:
        rig_data = json.load(f)
    
    joint_names = list(rig_data[rig_name].keys())
    marker_idxs = [[marker_labels.index(label) for label in rig_data[rig_name][joint]] for joint in joint_names]

    return joint_names, marker_idxs


def apply_rig_format(pose3d: np.ndarray, joint_names: List[str], marker_idxs: List[List[int]]) -> np.ndarray:
    formatted_pose3d = np.zeros((pose3d.shape[0], len(joint_names), 3))
    for i, idxs in enumerate(marker_idxs):
        formatted_pose3d[:, i, :] = np.mean(pose3d[:, idxs, :], axis=1)
    return formatted_pose3d


# def process_file(json_file: Path, rig_file: str, rig_name: str):
#     print(f"Converting {json_file} ...")
#     marker_data = get_marker_data(json_file)
#     time_range = get_time_range(marker_data)
#     pose3d, marker_labels = get_pose_array(marker_data, time_range)
#     joint_names, marker_idxs = load_rig_mapping(rig_file, rig_name, marker_labels)
#     formatted_pose3d = apply_rig_format(pose3d, joint_names, marker_idxs)
    
#     # Prepare output directory
#     dir_parts = list(json_file.parent.parts)
#     dir_parts = ['npy' if dp == 'json' else dp for dp in dir_parts]
#     output_dir = Path(*dir_parts)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Save the formatted pose3d array
#     output_filename = json_file.with_suffix('.npy').name
#     np.save(output_dir / output_filename, formatted_pose3d)

# NEW VERSION: directly save the features and labels in the proper folders 
def process_file(json_file: Path, rig_file: str, rig_name: str):
    print(f"Converting {json_file} ...")
    marker_data = get_marker_data(json_file)
    time_range = get_time_range(marker_data)
    pose3d, marker_labels = get_pose_array(marker_data, time_range)
    joint_names, marker_idxs = load_rig_mapping(rig_file, rig_name, marker_labels)
    formatted_pose3d = apply_rig_format(pose3d, joint_names, marker_idxs) # shape (T, 17, 3)
    # Now we have to adapt the dataset to the shape for the FACT model
    formatted_pose3d = formatted_pose3d.reshape(formatted_pose3d.shape[0], -1) # --> shape (T, 51)

    # Extract label from the folder name (e.g., 'Flip')
    label_str = json_file.parent.name
    if label_str not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label_str}")
    label = LABEL_MAP[label_str]
    frame_labels = np.full((formatted_pose3d.shape[0],), label, dtype=np.int64) # Expand the name of the label of the jump to all the frames
    # Check eventually mismatch
    T = min(formatted_pose3d.shape[0], frame_labels.shape[0])
    formatted_pose3d = formatted_pose3d[:T]
    frame_labels = frame_labels[:T]

    # Output filenames
    filename = json_file.with_suffix('.npy').name.replace('.npy', '')

    np.save(FEATURES_DIR / f"{filename}.npy", formatted_pose3d)
    # print(formatted_pose3d)
    np.save(LABELS_DIR / f"{filename}.npy", frame_labels)
    # print(frame_labels)

    return filename


def main():
    # parser
    parser = ArgumentParser()
    parser.add_argument("--rig", type=str, default='Human3.6M', help="Rig mapping to use")
    args = parser.parse_args()

    files = [Path(f'./json/{skater}/{jump}/{f}')
             for skater in SKATERS
             for jump in JUMPS
             for f in os.listdir(f'./json/{skater}/{jump}')
             if f.endswith('.json')]

    rig_file = './utils/rig.json'
    
    formatted_files = [process_file(json_file, rig_file, args.rig) for json_file in files]

    # Split percentages
    train_ratio = 0.7
    val_ratio = 0.15  # The remaining 15% will be for testing

    # Shuffle the dataset for randomness
    random.shuffle(formatted_files)

    # Prepare split indices
    num_samples = len(formatted_files)
    train_split = int(num_samples * train_ratio)
    val_split = train_split + int(num_samples * val_ratio)

    # Split into train, val, test
    train_files = formatted_files[:train_split]
    val_files = formatted_files[train_split:val_split]
    test_files = formatted_files[val_split:]

    # Save splits
    with open(os.path.join(SPLITS_DIR, "train.txt"), "w") as f:
        f.writelines("\n".join(train_files))
    with open(os.path.join(SPLITS_DIR, "val.txt"), "w") as f:
        f.writelines("\n".join(val_files))
    with open(os.path.join(SPLITS_DIR, "test.txt"), "w") as f:
        f.writelines("\n".join(test_files))

    print("Successfully converted and split all JSON files!")

if __name__ == '__main__':
    main()
