import os
import shutil
import tempfile
from pathlib import Path
import pytest
from core.indexer import CodeGraphIndexer
from core.universal_ingestion import UniversalIngestionPipeline

@pytest.fixture
def universal_pipeline():
    indexer = CodeGraphIndexer(collection_name="test_universal_col", location=":memory:")
    return UniversalIngestionPipeline(indexer=indexer)

def test_ingest_local_path(universal_pipeline):
    temp_dir = Path(tempfile.mkdtemp(prefix="test_local_repo_"))
    try:
        (temp_dir / "math_lib").mkdir()
        with open(temp_dir / "math_lib" / "ops.py", "w") as f:
            f.write("def multiply(x, y):\n    '''Moltiplica due valori'''\n    return x * y\n")

        stats = universal_pipeline.ingest(str(temp_dir))
        assert stats["source_type"] == "local_directory"
        assert stats["cpg_nodes"] >= 1
        assert stats["qdrant_points"] >= 1

        out_html = universal_pipeline.export_and_view_3d("test_universal_3d.html")
        assert Path(out_html).exists()
        Path(out_html).unlink()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_ingest_dataset_by_name(universal_pipeline):
    stats = universal_pipeline.ingest("openai/openai_humaneval", max_samples=3)
    assert stats["source_type"] == "huggingface_dataset"
    assert stats["cpg_nodes"] >= 3

def test_detect_github_url(universal_pipeline):
    assert universal_pipeline._is_github_url("https://github.com/psf/requests.git") is True
    assert universal_pipeline._is_github_url("git@github.com:psf/requests.git") is True
    assert universal_pipeline._is_github_url("openai/openai_humaneval") is False
    assert universal_pipeline._is_local_path("openai/openai_humaneval") is False

