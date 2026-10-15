import os
from pathlib import Path
from core.indexer import CodeGraphIndexer
from core.dataset_ingestion import DatasetIngestionPipeline
from core.visualizer import Graph3DVisualizer

def test_dataset_ingestion_and_3d_export():
    indexer = CodeGraphIndexer(collection_name="test_ingest_3d_col", location=":memory:")
    pipeline = DatasetIngestionPipeline(indexer=indexer)

    # Ingerisci un campione di test rapido (es. 5 task)
    stats = pipeline.ingest_humaneval(split="test", max_samples=5)
    
    assert stats["total_tasks_ingested"] == 5
    assert stats["cpg_nodes"] >= 5
    assert stats["qdrant_points"] >= 5

    # Esporta in HTML 3D
    output_file = "test_3d_graph.html"
    out_path = pipeline.export_3d_graph(output_file)
    
    assert Path(out_path).exists()
    content = Path(out_path).read_text(encoding="utf-8")
    assert "ForceGraph3D" in content
    assert "Code Property Graph 3D" in content
    assert len(content) > 1000

    # Cleanup test output
    if Path(out_path).exists():
        Path(out_path).unlink()

