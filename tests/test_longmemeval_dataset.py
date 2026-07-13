import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_bench.dataset.longmemeval import LongMemEvalDataset


class LongMemEvalDatasetTests(unittest.TestCase):
    def test_zh_split_downloads_llm_translation_from_github_when_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            calls = []

            def fake_urlretrieve(url, path):
                calls.append((url, Path(path)))
                Path(path).write_bytes(b"fake gzip")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("memory_bench.dataset.longmemeval.dataset_cache_dir", return_value=cache_dir),
                patch("memory_bench.dataset.longmemeval.urllib.request.urlretrieve", side_effect=fake_urlretrieve),
                patch("builtins.print"),
            ):
                ds = LongMemEvalDataset()
                path = ds._data_path("zh")
                second_path = ds._data_path("zh")

            self.assertEqual(path, cache_dir / "longmemeval_s_cleaned.llm.zh.json.gz")
            self.assertEqual(second_path, path)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0][0],
                "https://raw.githubusercontent.com/Sanderhoff-alt/longmemeval-zh/main/"
                "datasets/longmemeval_s_cleaned.llm.zh.json.gz",
            )

    def test_zh_split_reuses_cached_llm_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cached_path = cache_dir / "longmemeval_s_cleaned.llm.zh.json.gz"
            cached_path.write_bytes(b"cached gzip")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("memory_bench.dataset.longmemeval.dataset_cache_dir", return_value=cache_dir),
                patch("memory_bench.dataset.longmemeval.urllib.request.urlretrieve") as urlretrieve,
            ):
                ds = LongMemEvalDataset()
                path = ds._data_path("zh")

            self.assertEqual(path, cached_path)
            urlretrieve.assert_not_called()

    def test_zh_split_env_path_skips_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local.zh.json.gz"

            with (
                patch.dict(os.environ, {"LONGMEMEVAL_ZH_DATA_PATH": str(local_path)}, clear=True),
                patch("memory_bench.dataset.longmemeval.urllib.request.urlretrieve") as urlretrieve,
            ):
                ds = LongMemEvalDataset()
                path = ds._data_path("zh")

            self.assertEqual(path, local_path)
            urlretrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
