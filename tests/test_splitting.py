from __future__ import annotations

import hashlib
import unittest

from poidh_detector.contracts import SampleRecord
from poidh_detector.splitting import SplitRatios, assign_group_splits


def _rows() -> list[SampleRecord]:
    rows: list[SampleRecord] = []
    for group_index in range(30):
        label = group_index % 2
        generator = "flux" if label else None
        for image_index in range(2):
            identity = f"{group_index}-{image_index}"
            rows.append(
                SampleRecord(
                    sample_id=f"sample:{identity}",
                    source_id="source-a" if group_index % 3 else "source-b",
                    upstream_path=f"data/{identity}.png",
                    local_path=f"images/{identity}.png",
                    label=label,
                    content_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                    provenance_group=f"group:{group_index}",
                    generator_family=generator,
                    content_type="scene",
                )
            )
    return rows


class SplittingTests(unittest.TestCase):
    def test_assignment_is_order_independent_and_group_disjoint(self) -> None:
        rows = _rows()
        ratios = SplitRatios(train=0.7, validation=0.2, calibration=0.1)
        forward = assign_group_splits(rows, ratios=ratios, seed="323")
        reverse = assign_group_splits(reversed(rows), ratios=ratios, seed="323")

        self.assertEqual(forward, reverse)
        split_by_group: dict[str, set[str]] = {}
        for sample_id, split in forward.items():
            row = next(row for row in rows if row.sample_id == sample_id)
            split_by_group.setdefault(row.provenance_group, set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))
        self.assertEqual(set(forward.values()), {"train", "validation", "calibration"})

    def test_seed_changes_assignment(self) -> None:
        rows = _rows()
        ratios = SplitRatios(train=0.7, validation=0.2, calibration=0.1)
        self.assertNotEqual(
            assign_group_splits(rows, ratios=ratios, seed="323"),
            assign_group_splits(rows, ratios=ratios, seed="324"),
        )

    def test_rejects_group_with_conflicting_labels(self) -> None:
        rows = _rows()
        conflicting = SampleRecord(
            sample_id="conflict",
            source_id="source-a",
            upstream_path="data/conflict.png",
            local_path="images/conflict.png",
            label=1,
            content_sha256="f" * 64,
            provenance_group=rows[1].provenance_group,
            generator_family="flux",
            content_type="scene",
        )
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            assign_group_splits(
                [*rows, conflicting],
                ratios=SplitRatios(0.7, 0.2, 0.1),
                seed="323",
            )

    def test_each_source_and_class_reaches_every_split(self) -> None:
        rows = _rows()
        assignments = assign_group_splits(
            rows, ratios=SplitRatios(0.7, 0.2, 0.1), seed="323"
        )

        for source in {row.source_id for row in rows}:
            for label in (0, 1):
                observed = {
                    assignments[row.sample_id]
                    for row in rows
                    if row.source_id == source and row.label == label
                }
                self.assertEqual(observed, {"train", "validation", "calibration"})


if __name__ == "__main__":
    unittest.main()
