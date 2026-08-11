import unittest

from tools.analyze_corrected_route_prefetch_oracle import (
    CATEGORIES,
    bounded_oracle_model,
    disagreement_count,
    evaluate_predictor,
    split_windows,
)


def synthetic_window(category, transaction_index, corpus_index, offset):
    routes = {}
    walls = {}
    for layer in range(1, 48):
        routes[layer] = tuple(
            tuple((offset + layer * 11 + position * 7 + expert) % 256 for expert in range(8))
            for position in range(8)
        )
        walls[layer] = 100.0
    return {
        "category": category,
        "transaction_index": transaction_index,
        "corpus_index": corpus_index,
        "proposal_wall_ms": 1000.0,
        "verification_wall_ms": 1000.0,
        "routes": routes,
        "layer_wall_ms": walls,
    }


class CorrectedRoutePrefetchOracleTests(unittest.TestCase):
    def setUp(self):
        self.windows = []
        index = 0
        for category_index, category in enumerate(CATEGORIES):
            for transaction in range(1, 9):
                self.windows.append(
                    synthetic_window(category, transaction, index, category_index * 31 + transaction)
                )
                index += 1

    def test_chronological_split_and_discrimination(self):
        calibration, holdout = split_windows(self.windows)
        self.assertEqual(len(calibration), 16)
        self.assertEqual(len(holdout), 16)
        discrimination = disagreement_count(calibration, holdout)
        self.assertEqual(discrimination["holdout_events"], 16 * 47 * 8)
        self.assertGreater(discrimination["events_where_controls_disagree"], 0)

    def test_future_oracle_is_exact_logically_but_bandwidth_bounded_physically(self):
        calibration, holdout = split_windows(self.windows)
        logical = evaluate_predictor("offline_future_oracle", calibration, holdout)["aggregate"]
        self.assertEqual(logical["recall_at_8"], 1.0)
        self.assertEqual(logical["precision"], 1.0)
        bounded = bounded_oracle_model(holdout)["aggregate"]
        self.assertLessEqual(bounded["prefetch_bandwidth_tax"], 0.25)
        self.assertGreater(bounded["optimistic_hidden_acquisition_ms"], 0.0)

    def test_previous_layer_control_is_not_an_oracle(self):
        calibration, holdout = split_windows(self.windows)
        metrics = evaluate_predictor(
            "previous_layer_same_position", calibration, holdout
        )["aggregate"]
        self.assertLess(metrics["recall_at_8"], 1.0)
        self.assertGreater(metrics["events_with_prediction"], 0)


if __name__ == "__main__":
    unittest.main()
