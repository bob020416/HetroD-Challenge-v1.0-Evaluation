from __future__ import annotations

import unittest

import torch

from hetrod_metrics.valid_region import points_in_valid_region


class HetrodValidRegionTests(unittest.TestCase):
    def test_polygon_query_is_boundary_inclusive_and_respects_holes(self):
        records = [
            {
                "exterior": torch.tensor(
                    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
                ),
                "holes": [
                    torch.tensor(
                        [[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0], [4.0, 4.0]]
                    )
                ],
            }
        ]
        points = torch.tensor(
            [
                [1.0, 1.0],
                [0.0, 5.0],
                [5.0, 5.0],
                [11.0, 5.0],
            ]
        )

        inside = points_in_valid_region(points, records, chunk_size=2)

        self.assertEqual(inside.tolist(), [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
