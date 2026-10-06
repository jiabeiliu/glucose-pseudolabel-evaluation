import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import preprocess_and_pca


class PreprocessingTests(unittest.TestCase):
    def test_shapes_and_unobserved_outcome_exclusion(self):
        rows = 40
        frame = pd.DataFrame({
            "Glucose": np.linspace(70, 180, rows),
            "BMI": np.linspace(20, 40, rows),
            "Age": np.arange(rows) + 25,
            "BloodPressure": np.linspace(60, 95, rows),
            "Outcome": np.arange(rows) % 2,
        })
        frame.loc[3, "BMI"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            with_outcome = Path(directory) / "with.csv"
            without_outcome = Path(directory) / "without.csv"
            frame.to_csv(with_outcome, index=False)
            frame.drop(columns=["Outcome"]).to_csv(without_outcome, index=False)
            first = preprocess_and_pca(with_outcome)
            second = preprocess_and_pca(without_outcome)
        self.assertEqual(first[0].shape, (32, 3))
        self.assertEqual(first[1].shape, (8, 3))
        self.assertTrue(np.isfinite(first[0].to_numpy()).all())
        self.assertTrue(np.isfinite(first[1].to_numpy()).all())
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)
        self.assertTrue(set(first[2]).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
