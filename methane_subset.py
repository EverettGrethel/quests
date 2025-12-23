import random


def count_frames(filename):
    count = 0
    with open(filename, "r") as f:
        for line in f:
            # If the line is just an integer (e.g., "5"), it's a frame header
            stripped = line.strip()
            if stripped.isdigit():
                count += 1
    return count


def stream_extxyz_frames(filename):
    """
    Generator that yields frames from an extxyz file.
    Each frame is returned as a list of strings (including newline characters).
    """
    with open(filename, "r") as f:
        while True:
            header = f.readline()
            if not header:  # EOF
                return
            natoms = int(header.strip())

            props = f.readline()
            if not props:
                return

            # read atom lines
            atom_lines = [f.readline() for _ in range(natoms)]
            if any(l == "" for l in atom_lines):
                return

            yield [header, props] + atom_lines


def sample_extxyz(input_file, output_file, sample_fraction=0.05, seed=123):
    """
    Randomly sample frames from an extxyz file.
    sample_fraction: keep this fraction (0–1) of frames.
    """
    random.seed(seed)

    with open(output_file, "w") as out:
        for frame in stream_extxyz_frames(input_file):
            if random.random() < sample_fraction:
                for line in frame:
                    out.write(line)


if __name__ == "__main__":
    # Example usage: keep 5% of frames
    input_xyz = "/home/grethel/dev/quests/examples/methane/methane.extxyz"
    output_xyz = "/home/grethel/dev/quests/examples/methane/methane_subset.extxyz"
    sample_fraction = 0.001  # keep 0.5%

    sample_extxyz(input_xyz, output_xyz, sample_fraction)
    input_n_frames = count_frames(input_xyz)
    output_n_frames = count_frames(output_xyz)
    print("Sampling complete → wrote:", output_xyz)
    print(f"input frames {input_n_frames}")
    print(f"output frames {output_n_frames}")
