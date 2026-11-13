"""Headset-free checks: run with `python -m unittest -v`."""

import contextlib
import importlib
import io
import unittest
from unittest.mock import patch

import numpy as np

import lab5


class SignalProcessingTests(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 250
        t = np.arange(500) / self.sample_rate
        self.alpha = np.sin(2 * np.pi * 10 * t)
        self.beta = np.sin(2 * np.pi * 20 * t)

    def test_known_beta_alpha_power_ratio(self):
        # Twice the amplitude means four times the power: beta/alpha should be 4.
        data = np.array([self.alpha + 2 * self.beta])
        original = data.copy()
        ratio = lab5._mean_beta_alpha_ratio(data, [0], self.sample_rate)
        self.assertAlmostEqual(ratio, 4.0, delta=0.03)
        np.testing.assert_array_equal(data, original)

    def test_average_of_channel_ratios_and_eeg_row_selection(self):
        # Ignore row 0; the selected EEG ratios are 4 and 0.25, averaging to 2.125.
        data = np.array([
            np.zeros(500),
            self.alpha + 2 * self.beta,
            2 * self.alpha + self.beta,
        ])
        ratio = lab5._mean_beta_alpha_ratio(data, [1, 2], self.sample_rate)
        self.assertAlmostEqual(ratio, 2.125, delta=0.03)

    def test_short_recording_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            lab5._mean_beta_alpha_ratio(np.zeros((1, 10)), [0], self.sample_rate)

    def test_flat_signal_cannot_produce_a_valid_ratio(self):
        with self.assertRaisesRegex(ValueError, "alpha power"):
            lab5._mean_beta_alpha_ratio(np.zeros((1, 500)), [0], self.sample_rate)


class SessionTests(unittest.TestCase):
    def test_session_released_after_processing_error_or_interrupt(self):
        for error in (ValueError("processing failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with patch.object(lab5, "BoardShim") as shim, \
                     patch.object(lab5.time, "sleep"), \
                     patch.object(lab5, "_mean_beta_alpha_ratio", side_effect=error):
                    shim.get_board_descr.return_value = {
                        "sampling_rate": 250, "eeg_channels": [1, 2]
                    }
                    board = shim.return_value
                    board.is_prepared.return_value = True
                    with self.assertRaises(type(error)):
                        lab5.calculate_alpha_beta_ratio("test-port")
                    board.release_session.assert_called_once()

    def test_import_is_quiet_and_does_not_open_a_session(self):
        output = io.StringIO()
        with patch("brainflow.board_shim.BoardShim") as shim, \
             contextlib.redirect_stdout(output):
            importlib.reload(lab5)
            shim.assert_not_called()
        importlib.reload(lab5)  # Restore the real BoardShim after the patch.
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
