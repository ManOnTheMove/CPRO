import csv
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeDataFrame(self.rows[item])
        raise TypeError(item)

    def to_csv(self, path, index=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=FakeDataFrame))

RAW2CSV_PATH = Path(__file__).resolve().parents[1] / "data_preparation" / "raw2csv.py"
spec = importlib.util.spec_from_file_location("raw2csv", RAW2CSV_PATH)
raw2csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw2csv)


def write_jsonl(path, records):
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def read_csv(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def medqa_record(split):
    return {
        "question": f"{split} question",
        "answer": f"{split} answer",
        "options": {"A": f"{split} option A", "B": f"{split} option B"},
        "answer_idx": "A",
    }


def medmcqa_record(split):
    return {
        "question": f"{split} medmcqa question",
        "opa": f"{split} option A",
        "opb": f"{split} option B",
        "opc": f"{split} option C",
        "opd": f"{split} option D",
        "cop": 2,
        "choice_type": "single",
    }


def run_main(tmp_path, dataset_name, *extra_args):
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with mock.patch.object(sys, "argv", ["raw2csv.py", "--dataset_name", dataset_name, *extra_args]):
            raw2csv.main()
    finally:
        os.chdir(old_cwd)


class TestRaw2Csv(unittest.TestCase):
    def test_medqa_main_uses_each_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "data" / "raw"
            raw_dir.mkdir(parents=True)
            for split in ["train", "dev", "test"]:
                write_jsonl(raw_dir / f"medqa_{split}.jsonl", [medqa_record(split)])

            run_main(tmp_path, "medqa", "--sft_grpo_split")

            for split in ["train", "dev", "test"]:
                rows = read_csv(tmp_path / "data" / "processed" / f"medqa_{split}.csv")
                self.assertEqual(rows[0]["answer"], f"{split} answer")
                self.assertTrue(rows[0]["ground_truth"].startswith(f"{split} answer<SEP>{split}0<SEP>"))

            self.assertTrue((tmp_path / "data" / "processed" / "medqa_train_sft.csv").exists())
            self.assertTrue((tmp_path / "data" / "processed" / "medqa_train_grpo.csv").exists())

    def test_medmcqa_main_uses_medmcqa_raw_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "data" / "raw"
            raw_dir.mkdir(parents=True)
            for split in ["train", "dev"]:
                write_jsonl(raw_dir / f"medmcqa_{split}.json", [medmcqa_record(split)])

            run_main(tmp_path, "medmcqa")

            for split in ["train", "dev"]:
                rows = read_csv(tmp_path / "data" / "processed" / f"medmcqa_{split}.csv")
                self.assertTrue(rows[0]["question"].startswith(f"{split} medmcqa question"))
                self.assertEqual(rows[0]["correct_answer"], "B")
                self.assertEqual(rows[0]["answer"], f"{split} option B")


if __name__ == "__main__":
    unittest.main()
