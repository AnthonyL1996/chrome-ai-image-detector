import unittest

from poidh_benchmark.manifest import (
    MirageRow,
    select_balanced_generator_strata,
    select_balanced_strata,
)


class MirageManifestTests(unittest.TestCase):
    def test_selects_exact_balanced_count_per_content_type(self) -> None:
        rows = [
            MirageRow(
                file_name=f"{content_type}/{label}/{generator}/image-{index}.jpg",
                label=label,
                content_type=content_type,
            )
            for content_type in ("Human", "Object")
            for label, generator in (("0_real", "camera"), ("1_fake", "Flux"))
            for index in range(5)
        ]

        selected = select_balanced_strata(
            rows,
            content_types=("Human", "Object"),
            per_class_per_content=3,
            seed="323",
        )

        self.assertEqual(len(selected), 12)
        for content_type in ("Human", "Object"):
            for label in ("0_real", "1_fake"):
                self.assertEqual(
                    sum(
                        row.content_type == content_type and row.label == label
                        for row in selected
                    ),
                    3,
                )

    def test_selection_is_order_independent_and_seeded(self) -> None:
        rows = [
            MirageRow(
                file_name=f"Human/{label}/{generator}/image-{index}.jpg",
                label=label,
                content_type="Human",
            )
            for label, generator in (("0_real", "camera"), ("1_fake", "Flux"))
            for index in range(12)
        ]

        selected = select_balanced_strata(
            rows,
            content_types=("Human",),
            per_class_per_content=4,
            seed="323",
        )
        reversed_selected = select_balanced_strata(
            reversed(rows),
            content_types=("Human",),
            per_class_per_content=4,
            seed="323",
        )
        other_seed = select_balanced_strata(
            rows,
            content_types=("Human",),
            per_class_per_content=4,
            seed="different",
        )

        self.assertEqual(selected, reversed_selected)
        self.assertNotEqual(selected, other_seed)

    def test_rejects_incomplete_strata(self) -> None:
        rows = [
            MirageRow(
                file_name=f"Human/0_real/camera/image-{index}.jpg",
                label="0_real",
                content_type="Human",
            )
            for index in range(3)
        ]

        with self.assertRaisesRegex(ValueError, "incomplete stratum"):
            select_balanced_strata(
                rows,
                content_types=("Human",),
                per_class_per_content=2,
                seed="323",
            )

    def test_explicit_content_types_exclude_unpaired_anime(self) -> None:
        paired = [
            MirageRow(
                file_name=f"Human/{label}/{generator}/image.jpg",
                label=label,
                content_type="Human",
            )
            for label, generator in (("0_real", "camera"), ("1_fake", "Flux"))
        ]
        anime = MirageRow(
            file_name="Anime/1_fake/sd3.5/image.jpg",
            label="1_fake",
            content_type="Anime",
        )

        selected = select_balanced_strata(
            [*paired, anime],
            content_types=("Human",),
            per_class_per_content=1,
            seed="323",
        )

        self.assertEqual(selected, paired)

    def test_rejects_duplicate_file_names(self) -> None:
        duplicate = MirageRow(
            file_name="Human/0_real/image.jpg",
            label="0_real",
            content_type="Human",
        )

        with self.assertRaisesRegex(ValueError, "duplicate file_name"):
            select_balanced_strata(
                [duplicate, duplicate],
                content_types=("Human",),
                per_class_per_content=1,
                seed="323",
            )

    def test_rejects_invalid_quota_types(self) -> None:
        row = MirageRow(
            file_name="Human/0_real/image.jpg",
            label="0_real",
            content_type="Human",
        )

        for quota in (0, -1, 2.5, "2", True):
            with self.subTest(quota=quota):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    select_balanced_strata(
                        [row],
                        content_types=("Human",),
                        per_class_per_content=quota,
                        seed="323",
                    )

    def test_rejects_empty_or_duplicate_content_types(self) -> None:
        for content_types in ((), ("Human", "Human"), ("",)):
            with self.subTest(content_types=content_types):
                with self.assertRaises(ValueError):
                    select_balanced_strata(
                        [],
                        content_types=content_types,
                        per_class_per_content=1,
                        seed="323",
                    )

    def test_derives_generator_family_without_treating_real_as_generator(self) -> None:
        fake = MirageRow(
            file_name="Human/1_fake/Flux_xhs_v2/img_919.jpg",
            label="1_fake",
            content_type="Human",
        )
        real = MirageRow(
            file_name="Human/0_real/16a6857.jpg",
            label="0_real",
            content_type="Human",
        )

        self.assertEqual(fake.generator_family, "Flux_xhs_v2")
        self.assertIsNone(real.generator_family)

    def test_generator_stratified_selection_balances_fake_families(self) -> None:
        rows = [
            MirageRow(
                file_name=f"{content_type}/0_real/real-{index}.jpg",
                label="0_real",
                content_type=content_type,
            )
            for content_type in ("Animal", "Human")
            for index in range(8)
        ]
        rows.extend(
            MirageRow(
                file_name=f"{content_type}/1_fake/{generator}/fake-{index}.jpg",
                label="1_fake",
                content_type=content_type,
            )
            for content_type in ("Animal", "Human")
            for generator in ("Flux", "sd3.5", "Digicam")
            for index in range(4)
        )

        selected = select_balanced_generator_strata(
            rows,
            content_types=("Animal", "Human"),
            fake_generators=("Flux", "sd3.5", "Digicam"),
            per_class_per_content=5,
            seed="323",
        )

        self.assertEqual(len(selected), 20)
        self.assertEqual(sum(row.label == "0_real" for row in selected), 10)
        fake_counts = {
            generator: sum(row.generator_family == generator for row in selected)
            for generator in ("Flux", "sd3.5", "Digicam")
        }
        self.assertEqual(fake_counts, {"Flux": 4, "sd3.5": 3, "Digicam": 3})

    def test_generator_stratified_selection_rejects_missing_family(self) -> None:
        rows = [
            MirageRow("Human/0_real/real-1.jpg", "0_real", "Human"),
            MirageRow("Human/0_real/real-2.jpg", "0_real", "Human"),
            MirageRow("Human/1_fake/Flux/fake.jpg", "1_fake", "Human"),
        ]

        with self.assertRaisesRegex(ValueError, "incomplete generator stratum"):
            select_balanced_generator_strata(
                rows,
                content_types=("Human",),
                fake_generators=("Flux", "sd3.5"),
                per_class_per_content=2,
                seed="323",
            )


if __name__ == "__main__":
    unittest.main()
