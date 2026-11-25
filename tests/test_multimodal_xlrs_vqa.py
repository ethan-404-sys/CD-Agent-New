from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock

from PIL import Image
import pyarrow as pa

from experiments.multimodal_probe import xlrs_fetch_cd_subset as fetcher
from experiments.multimodal_probe import xlrs_vqa_run as vqa


class FetchTests(unittest.TestCase):
    def test_scans_arrow_metadata_and_addresses_only_target_images(self):
        distractor = b"not-a-target-image" * 400
        first, second = b"year-one-image" * 100, b"year-two-image" * 100
        batch = pa.RecordBatch.from_pylist([
            {"path": "plain", "index": 1, "question": "Not a change question",
             "multi-choice options": ["A", "B"], "answer": "A",
             "category": "Object properties/Object color", "l2-category": "default",
             "image": [{"bytes": distractor, "path": "irrelevant.tif"}]},
            {"path": "scene", "index": 42, "question": "What changed?",
             "multi-choice options": ["(A) Built", "(B) Removed"], "answer": "B",
             "category": "Counting/Counting with changing detection", "l2-category": "default",
             "image": [{"bytes": first, "path": "t1.tif"}, {"bytes": second, "path": "t2.tif"}]},
        ], schema=fetcher.SCHEMA)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, fetcher.SCHEMA) as writer:
            writer.write_batch(batch)
        payload = sink.getvalue().to_pybytes()
        remote = Mock(size=len(payload))
        remote.read.side_effect = lambda start, length: payload[start:start + length]
        rows, counts = fetcher.index_shard(remote)
        self.assertEqual(sum(counts.values()), 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].row["id"], "xlrs-42")
        self.assertEqual(rows[0].row["answer"], "B")
        self.assertEqual(rows[0].spans, ((payload.index(first), len(first)), (payload.index(second), len(second))))
        start, end = payload.index(distractor), payload.index(second) + len(second)
        for record in remote.read.call_args_list:
            a, length = record.args
            self.assertTrue(a + length <= start or a >= end)

    def test_reject_full_object_response_and_enforce_strict_byte_cap(self):
        budget = fetcher.Budget()
        with self.assertRaisesRegex(ValueError, "3 GB"):
            budget.reserve(fetcher.CAP)
        session = MagicMock()
        response = session.get.return_value.__enter__.return_value
        response.status_code = 200
        remote = fetcher.RemoteArrow(session, "https://example.invalid/file", 1000, budget)
        with self.assertRaisesRegex(ValueError, "exact Arrow range"):
            remote.read(0, 12)
        response.raw.read.assert_not_called()
        self.assertEqual(budget.transferred, 12)


class VQATests(unittest.TestCase):
    def test_single_and_paired_native_crops_and_gt_isolation(self):
        for copies in (1, 2):
            with self.subTest(copies=copies), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                photos = []
                for i in range(copies):
                    path = root / f"{i}.png"
                    Image.new("RGB", (1600, 1000), ("red", "blue")[i]).save(path)
                    photos.append(path.name)
                manifest = root / "dev_cd_vqa.jsonl"
                manifest.write_text(json.dumps({
                    "id": "scene", "category": "Change detection", "question": "What <changed>?",
                    "options": ["(A) Houses", "(B) Roads", "(C) Trees", "(D) Nothing"], "answer": "C",
                    "images": photos, "image_sizes": [[1600, 1000]] * copies,
                    "secret": "GT_ISOLATION_SENTINEL",
                }), encoding="utf-8")
                [question], answers = vqa.read_questions(manifest)
                self.assertFalse(hasattr(question, "answer"))
                replies = iter([
                    "B", '{"tool":"view","region":[1100,50,1500,250]}',
                    '{"tool":"submit_answer","choice":"B","evidence_regions":[[1100,50,1500,250]],'
                    '"rationale":"Road in the crop"}',
                ])
                calls = []

                def call(messages, *, max_tokens):
                    captured = json.loads(json.dumps(messages))
                    calls.append(captured)
                    serialized = json.dumps(captured)
                    self.assertNotIn("GT_ISOLATION_SENTINEL", serialized)
                    self.assertNotIn('"answer"', serialized)
                    return next(replies), {"prompt_tokens": 200, "completion_tokens": 12}

                viz = root / "viz"
                asset_dir = viz / "assets" / question.id
                runs = vqa.run_question(question, asset_dir, ("global_only", "zoom_agent"), call)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(calls[0]), 1)
                self.assertEqual(len(calls[1][0]["content"]), copies + 1)
                self.assertEqual(len(calls[2][-1]["content"]), copies + 1)
                self.assertEqual("scan_diff_hotspots" in calls[1][0]["content"][0]["text"], copies == 2)
                result = runs["zoom_agent"]
                self.assertEqual(result.choice, "B")
                self.assertEqual(result.zooms[0]["region"], (1100, 50, 1500, 250))
                self.assertEqual(len(result.zooms[0]["files"]), copies)
                for i, filename in enumerate(result.zooms[0]["files"]):
                    with Image.open(asset_dir / filename) as image:
                        self.assertEqual(image.size, (400, 200))
                        self.assertGreater(image.getpixel((150, 100))[2 * i], 245)
                with Image.open(asset_dir / "global-1.jpg") as image:
                    self.assertEqual(max(image.size), 1024)
                row = dict(id=question.id, category=question.category, question=question.question,
                           options=question.options, image_sizes=question.image_sizes, answer=answers[question.id],
                           runs={mode: asdict(run) for mode, run in runs.items()})
                metrics = vqa.score([row], tuple(runs))
                self.assertEqual(metrics["modes"]["zoom_agent"]["average_zoom_steps"], 1)
                self.assertEqual(metrics["modes"]["global_only"]["accuracy"], 0)
                self.assertGreater(metrics["total_cost_cny"], 0)
                vqa.write_viz(viz, [row], metrics)
                page = (viz / "index.html").read_text()
                self.assertIn('What &lt;changed&gt;?', page)
                self.assertIn("GT: <b>C</b>", page)
                self.assertIn("overlay-1.jpg", page)
                self.assertIn("step-1-image-1.jpg", page)

    def test_budget_does_not_allow_ninth_reply_or_final_view(self):
        q = vqa.Question("x", "Change detection", "Question?", ("a", "b"), (), ((200, 200),))
        native = [Image.new("RGB", (200, 200))]
        call = Mock(return_value=('{"tool":"grid_overview","rows":2,"cols":2}', None))
        result = vqa.answer_zoom(q, native, native, [], Path("."), call)
        self.assertEqual(call.call_count, 8)
        self.assertEqual(result.replies, 8)
        self.assertIsNone(result.choice)
        self.assertIn("reply budget exhausted", result.trace[-1]["error"])

    def test_diff_hotspots_and_exact_choices(self):
        a, b = Image.new("RGB", (256, 256)), Image.new("RGB", (256, 256))
        b.paste("white", (224, 224, 256, 256))
        self.assertEqual(vqa.diff_hotspots([a, b], (8000, 8000), 1), [[7000, 7000, 8000, 8000]])
        self.assertEqual(vqa.parse_choice("BA", "ABCD"), "AB")
        self.assertIsNone(vqa.parse_choice("Answer: A", "ABCD"))
        self.assertIsNone(vqa.parse_choice("D", "ABC"))
        with self.assertRaises(ValueError):
            vqa.validate_box([0, 0, 8001, 8000], (8000, 8000))


if __name__ == "__main__":
    unittest.main()
