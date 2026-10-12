import unittest

from prepare_common_range_probe import frame_ids, select_cases


class CommonRangeProbePlanTests(unittest.TestCase):
    def inputs(self):
        rows, manifests, old_ids = [], {"train": {}, "valid": {}}, set()
        for split, number, maxabs in (("train", 23, 5.), ("valid", 8, 3.)):
            for index in range(number):
                for kind, extent in (("exposed", maxabs), ("control", 1.2)):
                    asset = f"{split}_{index}_{kind}"
                    rows.append({
                        "asset_id": asset, "split": split, "path": "/npz/" + asset + ".npz", "status": "ok",
                        "integration_source": "base" if index % 2 else "addition", "vertex_count": 1000 + index,
                        "frame_count": 40, "manifest_row_index": len(rows),
                        "eligible_raw_stored_frames": list(range(32)), "heldout_raw_stored_frames": list(range(32, 40)),
                        "eligible_both_banks": {"maxabs": extent, "worst_envelope": {
                            "reference_raw_stored_frame": 7, "raw_stored_frame": 29}},
                    })
                    manifests[split][asset] = {"asset_id": asset, "path": rows[-1]["path"],
                                              "canonical_metrics": {"target_joint_count": index + 4}}
                    if split == "train" and index % 2:
                        old_ids.add(asset)
        return rows, manifests, old_ids

    def test_pairs_match_and_include_real_extreme(self):
        result = select_cases(*self.inputs())
        self.assertEqual(len(result), 62)
        self.assertEqual(len({row["asset_id"] for row in result}), 62)
        for exposed, control in zip(result[::2], result[1::2]):
            for key in ("split", "integration_source", "old_train_member", "pair_index", "target_joint_count"):
                self.assertEqual(exposed[key], control[key])
            self.assertEqual(exposed["selected_frames"][:2], [7, 29])
            self.assertEqual(len(set(exposed["selected_frames"])), 8)
            self.assertLessEqual(control["audited_maxabs"], 1.5)

    def test_changed_source_path_fails(self):
        rows, manifests, old_ids = self.inputs()
        rows[0]["path"] = "/wrong.npz"
        with self.assertRaisesRegex(ValueError, "paths differ"):
            select_cases(rows, manifests, old_ids)

    def test_heldout_witness_fails(self):
        row = self.inputs()[0][0]
        row["eligible_both_banks"]["worst_envelope"]["raw_stored_frame"] = 36
        with self.assertRaisesRegex(ValueError, "eligible-frame"):
            frame_ids(row)

    def test_missing_control_does_not_cross_split(self):
        rows, manifests, old_ids = self.inputs()
        rows = [row for row in rows if not (row["split"] == "valid" and row["integration_source"] == "addition"
                                          and row["eligible_both_banks"]["maxabs"] < 2)]
        with self.assertRaisesRegex(ValueError, "matched control"):
            select_cases(rows, manifests, old_ids)


if __name__ == "__main__":
    unittest.main()
