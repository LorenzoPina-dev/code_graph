import os
import shutil
import tempfile
from pathlib import Path
from core.indexer import CodeGraphIndexer
from core.ingestion import RepositoryIngestionPipeline

def test_ingestion():
    print("Testing RepositoryIngestionPipeline with .gitignore filtering...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_repo_"))
    
    try:
        # Crea struttura directory
        (temp_dir / "services").mkdir()
        (temp_dir / "ignored_dir").mkdir()
        (temp_dir / ".git").mkdir()

        # .gitignore
        with open(temp_dir / ".gitignore", "w") as f:
            f.write("ignored_dir/\n*.secret.py\n")

        # File validi
        with open(temp_dir / "services" / "auth.py", "w") as f:
            f.write("def login(user, pwd):\n    '''Esegue login utente.'''\n    return True\n")

        with open(temp_dir / "services" / "db.py", "w") as f:
            f.write("def connect():\n    return 'connected'\n")

        # File ignorati
        with open(temp_dir / "ignored_dir" / "secret.py", "w") as f:
            f.write("SECRET = '12345'\n")

        with open(temp_dir / "keys.secret.py", "w") as f:
            f.write("API_KEY = 'secret'\n")

        # File binario/non python
        with open(temp_dir / "image.png", "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        indexer = CodeGraphIndexer(collection_name="ingestion_test_col", location=":memory:")
        pipeline = RepositoryIngestionPipeline(indexer)
        summary = pipeline.scan_and_index_repository(str(temp_dir))

        print(f"Ingestion summary: {summary}")

        assert summary["indexed_files_count"] == 2, f"Expected 2 indexed files, got {summary['indexed_files_count']}"
        assert any("auth.py" in f for f in summary["indexed_files"])
        assert any("db.py" in f for f in summary["indexed_files"])
        assert not any("secret" in f for f in summary["indexed_files"])
        assert not any("image.png" in f for f in summary["indexed_files"])

        print("TASK 6 VALIDATION SUCCESSFUL: Repository Ingestion pipeline correctly filtered files and indexed repository!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_ingestion()

