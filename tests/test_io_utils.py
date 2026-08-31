"""Tests for surviving an abrupt power loss mid-write.

Long runs append a row per question over many hours. If the machine loses
power partway through an append, the file ends in a half-written line. Every
resume path reads its own output file to work out where it stopped, so an
unrepaired torn line turns a recoverable interruption into a crash, and the
tempting fix -- deleting the file -- throws away hours of completed work.

The torn row is dropped rather than guessed at: that question simply gets
answered again on the next run.
"""

import json
import tempfile
import unittest
from pathlib import Path

from thesis_pipeline.io_utils import append_jsonl, read_jsonl, repair_jsonl


class RepairJsonlTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "rows.jsonl"

    def tearDown(self):
        self._dir.cleanup()

    def write_raw(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")

    def test_intact_file_is_left_alone(self):
        for i in range(3):
            append_jsonl(self.path, {"id": i})
        before = self.path.read_text(encoding="utf-8")
        self.assertEqual(repair_jsonl(self.path), 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_half_written_final_line_is_dropped(self):
        self.write_raw('{"id": 0}\n{"id": 1}\n{"id": 2, "predi')
        self.assertEqual(repair_jsonl(self.path), 1)
        self.assertEqual([row["id"] for row in read_jsonl(self.path)], [0, 1])

    def test_repaired_file_is_strictly_readable_again(self):
        # The point of rewriting rather than skipping on read: the run appends
        # more rows afterwards, and analyze_supervision.py reads the result
        # strictly. A tolerated bad line would sit in the middle of the file
        # and resurface hours later.
        self.write_raw('{"id": 0}\n{"id": 1, "trunc')
        repair_jsonl(self.path)
        append_jsonl(self.path, {"id": 1})
        self.assertEqual([row["id"] for row in read_jsonl(self.path)], [0, 1])

    def test_nul_padding_from_an_unflushed_write_is_dropped(self):
        # NTFS can extend a file's length before its data reaches disk, so a
        # power cut can leave a run of NUL bytes rather than a clean prefix.
        self.write_raw('{"id": 0}\n' + "\x00" * 64)
        self.assertEqual(repair_jsonl(self.path), 1)
        self.assertEqual([row["id"] for row in read_jsonl(self.path)], [0])

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(repair_jsonl(self.path / "nope.jsonl"), 0)

    def test_completely_empty_file_is_not_an_error(self):
        self.write_raw("")
        self.assertEqual(repair_jsonl(self.path), 0)

    def test_corruption_before_the_end_is_also_dropped(self):
        # Not just the last line: an interrupted flush can leave a gap in the
        # middle. Dropping id 1 means it gets recomputed, which is correct.
        self.write_raw('{"id": 0}\n{"bro' + '\n{"id": 2}\n')
        self.assertEqual(repair_jsonl(self.path), 1)
        self.assertEqual([row["id"] for row in read_jsonl(self.path)], [0, 2])

    def test_original_is_preserved_when_nothing_is_wrong(self):
        # Repair must not rewrite a healthy file: rewriting is itself a window
        # in which power can be lost.
        append_jsonl(self.path, {"id": 0})
        mtime = self.path.stat().st_mtime_ns
        repair_jsonl(self.path)
        self.assertEqual(self.path.stat().st_mtime_ns, mtime)


if __name__ == "__main__":
    unittest.main()
