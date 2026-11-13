"""ECE441 Lab 5: acquire EEG and return the mean per-channel beta/alpha ratio."""

import argparse
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations


# Replace this with your dongle's actual path from `ls /dev/tty.*` on macOS.
# The game calls the function without arguments, so it also uses this default.
DEFAULT_PORT = "/dev/tty.usbserial-REPLACE_ME"


def _mean_beta_alpha_ratio(data, eeg_channels, sample_rate):
    """Process a captured data block; this helper does not access hardware."""
    nfft = DataFilter.get_nearest_power_of_two(sample_rate)
    if data.shape[1] < nfft:
        raise ValueError(
            f"Need at least {nfft} samples for Welch PSD; got {data.shape[1]}. "
            "Check the connection or allow more acquisition time."
        )

    ratios = np.zeros(len(eeg_channels))
    for index, channel in enumerate(eeg_channels):
        # BrainFlow modifies signals in place; copy this channel before detrending.
        signal = np.array(data[channel], dtype=np.float64, copy=True)
        if not np.all(np.isfinite(signal)):
            raise ValueError(f"EEG channel {channel} contains non-finite samples.")

        # Remove the baseline and linear drift to reduce leakage into the PSD.
        DataFilter.detrend(signal, DetrendOperations.LINEAR.value)

        # Welch averages overlapping spectra; a Hanning window reduces leakage.
        psd = DataFilter.get_psd_welch(
            signal, nfft, nfft // 2, sample_rate, WindowOperations.HANNING.value
        )
        alpha_power = DataFilter.get_band_power(psd, 7.0, 13.0)
        beta_power = DataFilter.get_band_power(psd, 14.0, 30.0)

        # Despite the function's required name, the handout specifies BETA / ALPHA.
        if alpha_power <= 0 or not np.isfinite(alpha_power):
            raise ValueError(f"EEG channel {channel} has invalid alpha power.")
        ratios[index] = beta_power / alpha_power
        if not np.isfinite(ratios[index]):
            raise ValueError(f"EEG channel {channel} has an invalid beta/alpha ratio.")

    # Average individual ratios, rather than dividing the two average powers.
    return float(np.mean(ratios))


def calculate_alpha_beta_ratio(port=DEFAULT_PORT):
    """Capture about two seconds of Cyton EEG and return one mean beta/alpha ratio."""
    if "REPLACE_ME" in port:
        raise ValueError(
            "Set DEFAULT_PORT in lab5.py to your dongle's actual serial port, "
            "or run this script with --port YOUR_PORT."
        )

    # Read board metadata instead of assuming a sampling rate or channel rows.
    board_id = BoardIds.CYTON_BOARD.value
    board_description = BoardShim.get_board_descr(board_id)
    sample_rate = int(board_description["sampling_rate"])
    eeg_channels = board_description["eeg_channels"]

    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        board.start_stream()
        time.sleep(2)  # Accumulate samples before estimating band power.
        data = board.get_board_data()
        return _mean_beta_alpha_ratio(data, eeg_channels, sample_rate)
    finally:
        # Free the serial connection even if processing fails or Ctrl+C is pressed.
        if board.is_prepared():
            board.release_session()


if __name__ == "__main__":
    # This block runs only as a script; importing from the game stays quiet.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT, help="OpenBCI dongle serial port")
    args = parser.parse_args()
    try:
        while True:
            ratio = calculate_alpha_beta_ratio(args.port)
            print(f"alpha-beta-ratio (beta/alpha): {ratio:.6f}", flush=True)
    except KeyboardInterrupt:
        pass  # Ctrl+C exits the loop after the function releases its session.
