import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("user_behavior_dataset.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("user_behavior_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):

        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):

        model = joblib.load(
            "user_behavior_model.pkl"
        )

        sample = pd.DataFrame([{
            "Device Model": "Google Pixel 5",
            "Operating System": "Android",
            "App Usage Time (min/day)": 250,
            "Screen On Time (hours/day)": 5.0,
            "Battery Drain (mAh/day)": 1200,
            "Number of Apps Installed": 50,
            "Data Usage (MB/day)": 800,
            "Age": 25,
            "Gender": "Male"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [1, 2, 3, 4, 5]
        )


if __name__ == "__main__":
    unittest.main()
