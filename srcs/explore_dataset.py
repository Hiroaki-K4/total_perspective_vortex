import mne
import argparse


def main(eeg_data_path: str):
    print("eeg_data_path: ", eeg_data_path)
    raw = mne.io.read_raw_edf(eeg_data_path)
    print(raw)
    print(raw.info)
    print(raw.info["bads"])
    raw.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_data_path")
    args = parser.parse_args()
    main(args.eeg_data_path)
