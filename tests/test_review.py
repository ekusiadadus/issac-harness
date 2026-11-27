import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from harness.project import file_hash, write_json
from harness.review import accept, compare, template


class VisualAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def capture(self, name, color):
        root = self.root / name
        (root / "normal").mkdir(parents=True)
        png = root / "normal/cam.png"
        Image.new("RGB", (12, 9), color).save(png)
        photo = root / "reference.jpg"
        Image.new("RGB", (12, 9), (120, 75, 32)).save(photo)
        frame = {"sha256": file_hash(png)}
        run = {
            "backend": "cycles",
            "status": "rendered",
            "cases": {"normal": {"renderer": {"name": "Cycles"}, "frames": {"cam": frame}}},
            "references": {"cam": {"file": "reference.jpg", "sha256": file_hash(photo)}},
        }
        write_json(root / "run.json", run)
        write_json(
            root / "check.json",
            {"status": "pass", "cases": {"normal": {"status": "pass", "views": {"cam": {}}}}},
        )
        template(root, SimpleNamespace(name="test"), run)
        return root

    def test_requires_an_observation_per_hash_bound_frame_and_reference(self):
        root = self.capture("first", "white")
        self.assertEqual(accept(root)["status"], "fail")
        review = root / "review.md"
        review.write_text(review.read_text().replace("| | 未確認 |", "| 試料の縁に折り目を確認 | 合格 |"))
        self.assertEqual(accept(root)["status"], "pass")
        with (root / "reference.jpg").open("ab") as out:
            out.write(b"changed")
        self.assertEqual(accept(root)["status"], "fail")
        self.assertEqual(json.loads((root / "acceptance.json").read_text())["status"], "fail")

    def test_measures_independent_golden_and_detects_pixel_regression(self):
        golden = self.capture("golden", "white")
        fresh = self.capture("replay", (250, 250, 250))
        self.assertEqual(compare(golden, fresh, 0.025)["status"], "pass")
        self.assertEqual(compare(golden, fresh, 0.01)["status"], "fail")
        with self.assertRaisesRegex(ValueError, "separate capture"):
            compare(golden, golden, 0.025)
