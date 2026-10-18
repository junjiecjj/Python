import os
import csv
import math
import numpy as np

# ✅ Torso positions (green dots)
torso_positions = {
    '1': (-1.365, -0.455), '2': (-0.455, -0.455), '3': (-0.455, -1.365), '4': (-1.365, -1.365),
    '5': (-0.91, -0.91), '6': (-2.275, -1.365), '7': (-2.275, -2.275), '8': (-1.365, -2.275)
}

# ✅ Receiver positions (red squares)
rx_positions = {
    '1': (0.0, -0.5), '2': (-(0.5 + 0.9), 0.5), '3': (-2.0, 0),
    '4': (0.5, -0.5), '5': (0.5, -(0.5 + 0.9)), '6': (0, -2.0)
}

# ✅ Transmitter position (blue triangle)
tx_position = (0, 0)

# ✅ Room mapping table
room_mapping_by_date = {
    '20181109': '1', '20181112': '1', '20181115': '1', '20181116': '1',
    '20181117': '2', '20181118': '2', '20181121': '1', '20181127': '2',
    '20181128': '2', '20181130': '1', '20181204': '2', '20181205_2': '2',
    '20181205_3': '2', '20181208': '2', '20181209': '2', '20181211': '3'
}

# ✅ Gesture mapping table
gesture_mapping_by_date = {
    '20181109': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Slide", "5": "Draw-Zigzag(Vertical)",
                 "6": "Draw-N(Vertical)"},
    '20181112': {"1": "Draw-1", "2": "Draw-2", "3": "Draw-3", "4": "Draw-4", "5": "Draw-5", "6": "Draw-6",
                 "7": "Draw-7", "8": "Draw-8", "9": "Draw-9", "10": "Draw-0"},
    '20181115': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Draw-O(Vertical)", "5": "Draw-Zigzag(Vertical)",
                 "6": "Draw-N(Vertical)"},
    '20181116': {"1": "Draw-1", "2": "Draw-2", "3": "Draw-3", "4": "Draw-4", "5": "Draw-5", "6": "Draw-6",
                 "7": "Draw-7", "8": "Draw-8", "9": "Draw-9", "10": "Draw-0"},
    '20181117': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Draw-O(Vertical)", "5": "Draw-Zigzag(Vertical)",
                 "6": "Draw-N(Vertical)"},
    '20181118': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Draw-O(Vertical)", "5": "Draw-Zigzag(Vertical)",
                 "6": "Draw-N(Vertical)"},
    '20181121': {"1": "Slide", "2": "Draw-O(Horizontal)", "3": "Draw-Zigzag(Horizontal)", "4": "Draw-N(Horizontal)",
                 "5": "Draw-Triangle(Horizontal)", "6": "Draw-Rectangle(Horizontal)"},
    '20181127': {"1": "Slide", "2": "Draw-O(Horizontal)", "3": "Draw-Zigzag(Horizontal)", "4": "Draw-N(Horizontal)",
                 "5": "Draw-Triangle(Horizontal)", "6": "Draw-Rectangle(Horizontal)"},
    '20181128': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Draw-O(Horizontal)", "5": "Draw-Zigzag(Horizontal)",
                 "6": "Draw-N(Horizontal)"},
    '20181130': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Slide", "5": "Draw-O(Horizontal)",
                 "6": "Draw-Zigzag(Horizontal)", "7": "Draw-N(Horizontal)", "8": "Draw-Triangle(Horizontal)",
                 "9": "Draw-Rectangle(Horizontal)"},
    '20181204': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Slide", "5": "Draw-O(Horizontal)",
                 "6": "Draw-Zigzag(Horizontal)", "7": "Draw-N(Horizontal)", "8": "Draw-Triangle(Horizontal)",
                 "9": "Draw-Rectangle(Horizontal)"},
    '20181205_2': {"1": "Draw-O(Horizontal)", "2": "Draw-Zigzag(Horizontal)", "3": "Draw-N(Horizontal)",
                   "4": "Draw-Triangle(Horizontal)", "5": "Draw-Rectangle(Horizontal)"},
    '20181205_3': {"1": "Slide", "2": "Draw-O(Horizontal)", "3": "Draw-Zigzag(Horizontal)", "4": "Draw-N(Horizontal)",
                   "5": "Draw-Triangle(Horizontal)", "6": "Draw-Rectangle(Horizontal)"},
    '20181208': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Slide"},
    '20181209': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Slide", "5": "Draw-O(Horizontal)",
                 "6": "Draw-Zigzag(Horizontal)"},
    '20181211': {"1": "Push&Pull", "2": "Sweep", "3": "Clap", "4": "Slide", "5": "Draw-O(Horizontal)",
                 "6": "Draw-Zigzag(Horizontal)"}
}


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_diff_distance(torso, rx, tx=tx_position):
    d_tx = euclidean(tx, torso)
    d_rx = euclidean(rx, torso)
    d_txrx = euclidean(tx, rx)
    return round((d_tx + d_rx - d_txrx), 6)


def parse_filename(filename):
    """
    Filename format: id-gesture-torso-face-rep-Rx.npy
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) != 6 or ext.lower() != '.npy':
        raise ValueError(f"Invalid filename format: {filename}")
    return {
        'id': parts[0],
        'gesture': parts[1],
        'torso_location': parts[2],
        'face_orientation': parts[3],
        'repetition': parts[4],
        'receiver': parts[5].lower().replace('r', '')
    }


def collect_and_write(root_dir, output_csv):
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"[Info] Removed existing file: {output_csv}")

    columns = ['FilePath', 'id', 'gesture_number', 'gesture_name',
               'torso_location', 'face_orientation', 'repetition',
               'receiver', 'room', 'diff_distance',
               'range_1', 'range_2', 'range_3']

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    file_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.npy'):
                full_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(full_path, root_dir)
                try:
                    info = parse_filename(file)
                    date = rel_path.split(os.sep)[0]
                    gesture_num = info['gesture']
                    gesture_name = gesture_mapping_by_date.get(date, {}).get(gesture_num, "Unknown")
                    room = room_mapping_by_date.get(date, "Unknown")

                    torso_loc = torso_positions.get(info['torso_location'])
                    rx_loc = rx_positions.get(info['receiver'])
                    diff_dist = compute_diff_distance(torso_loc, rx_loc) if torso_loc and rx_loc else None

                    data = np.load(full_path, allow_pickle=True).item()
                    range_vals = data.get('range', [None, None, None])
                    if isinstance(range_vals, np.ndarray):
                        range_vals = range_vals.tolist()

                    row = {
                        'FilePath': rel_path.replace('\\', '/'),
                        'id': info['id'],
                        'gesture_number': gesture_num,
                        'gesture_name': gesture_name,
                        'torso_location': info['torso_location'],
                        'face_orientation': info['face_orientation'],
                        'repetition': info['repetition'],
                        'receiver': info['receiver'],
                        'room': room,
                        'diff_distance': diff_dist,
                        'range_1': range_vals[0],
                        'range_2': range_vals[1],
                        'range_3': range_vals[2]
                    }

                    with open(output_csv, mode='a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=columns)
                        writer.writerow(row)

                    print(f"[OK] {rel_path} → diff = {diff_dist}")
                    file_count += 1
                except Exception as e:
                    print(f"[Skip] {file} → {e}")

    print(f"\n✅ Total saved: {file_count} samples → {output_csv}")


if __name__ == '__main__':
    root_dir = './'  # Root directory of the dataset
    output_csv = 'gesture_metadata.csv'
    collect_and_write(root_dir, output_csv)
