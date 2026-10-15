"""
Universal Ingestion Pipeline:
Ingerisce sorgenti di codice arbitrarie da 3 canali unificati:
1. Repository GitHub (es. 'https://github.com/user/repo.git' o 'https://github.com/user/repo')
   - Clona in directory di lavoro locale (shallow clone con depth=1)
   - Applica filtraggio .gitignore e scarto file binari
2. Percorso Locale (es. 'D:/progetti/mio_software' o './my_package')
   - Scansione ricorsiva della directory con rispetto di .gitignore
3. Nome Dataset da Libreria (es. 'openai/openai_humaneval', 'humaneval', 'mbpp', ecc.)
   - Scarica il dataset via Hugging Face datasets ed estrae il codice
In tutti i casi:
- Esegue parsing sintattico AST (Tree-sitter)
- Costruisce il Code Property Graph (NetworkX) con relazioni contains e calls
- Popola Qdrant (Dense + Sparse BM25)
- Genera il visualizzatore 3D interattivo pronto da aprire nel browser
"""

import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse
from datasets import load_dataset

from .chunker import ASTCodeChunker, CodeChunk
from .indexer import CodeGraphIndexer
from .ingestion import RepositoryIngestionPipeline
from .visualizer import Graph3DVisualizer

class UniversalIngestionPipeline:
    def __init__(self, indexer: Optional[CodeGraphIndexer] = None, chunker: Optional[ASTCodeChunker] = None):
        self.chunker = chunker or ASTCodeChunker()
        self.indexer = indexer or CodeGraphIndexer(collection_name="universal_code_graph", location=":memory:")
        self.repo_pipeline = RepositoryIngestionPipeline(indexer=self.indexer, chunker=self.chunker)

    def _is_github_url(self, source: str) -> bool:
        """Determina se la stringa fornita è un URL GitHub o Git valido."""
        s = source.strip()
        if s.startswith("git@github.com:") or "github.com/" in s or s.endswith(".git"):
            return True
        try:
            parsed = urlparse(s)
            return bool(parsed.scheme in ["http", "https", "git"] and "github" in parsed.netloc)
        except Exception:
            return False

    def _is_local_path(self, source: str) -> bool:
        """Determina se la stringa è un percorso di file system locale esistente."""
        try:
            p = Path(source).expanduser().resolve()
            return p.exists()
        except Exception:
            return False

    def ingest_github_repository(
        self,
        repo_url: str,
        target_dir: Optional[str] = None,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Clona un repository GitHub remoto ed esegue l'ingestione completa del codice e del grafo.
        """
        print(f"[*] Clonazione repository GitHub in corso da: {repo_url}...")
        is_temp = False
        if not target_dir:
            temp_base = Path(tempfile.gettempdir()) / "code_rag_repos"
            temp_base.mkdir(parents=True, exist_ok=True)
            # Normalizza nome dal repository
            repo_name = re.sub(r"[^A-Za-z0-9_-]", "_", repo_url.rstrip("/").split("/")[-1].replace(".git", ""))
            target_dir = str(temp_base / f"{repo_name}_{os.getpid()}")
            is_temp = True

        dest_path = Path(target_dir).resolve()
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)

        git_cmd = ["git", "clone", "--depth", "1"]
        if branch:
            git_cmd.extend(["-b", branch])
        git_cmd.extend([repo_url, str(dest_path)])

        try:
            res = subprocess.run(git_cmd, capture_output=True, text=True, check=True)
            print(f"[+] Repository clonato con successo in: {dest_path}")
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr or e.stdout
            raise RuntimeError(f"Errore durante git clone: {err_msg}")

        # Ingestione tramite RepositoryIngestionPipeline
        stats = self.repo_pipeline.scan_and_index_repository(str(dest_path))
        stats["source_type"] = "github_repository"
        stats["source"] = repo_url
        stats["cloned_path"] = str(dest_path)
        stats["qdrant_points"] = self.indexer.client.count(collection_name=self.indexer.collection_name).count
        return stats

    def ingest_local_path(self, local_path: str) -> Dict[str, Any]:
        """
        Ingerisce una cartella o progetto locale scansionando ricorsivamente con filtri .gitignore.
        """
        p = Path(local_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Percorso locale non trovato: {local_path}")

        print(f"[*] Ingestione percorso locale: {p}...")
        stats = self.repo_pipeline.scan_and_index_repository(str(p))
        stats["source_type"] = "local_directory"
        stats["source"] = str(p)
        stats["qdrant_points"] = self.indexer.client.count(collection_name=self.indexer.collection_name).count
        return stats

    def ingest_dataset_by_name(
        self,
        dataset_name: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Scarica ed ingerisce un dataset di codice noto o arbitrario da Hugging Face.
        Supporta 'openai_humaneval', 'openai/openai_humaneval', 'mbpp', o qualsiasi dataset con codice.
        """
        # Normalizzazione alias comuni
        canonical_name = dataset_name.strip()
        if canonical_name.lower() in ["humaneval", "openai_humaneval"]:
            canonical_name = "openai/openai_humaneval"

        print(f"[*] Download e ingestione dataset '{canonical_name}' (split: {split})...")
        try:
            dataset = load_dataset(canonical_name, split=split)
        except Exception:
            # Fallback a split='train' se 'test' non esiste
            dataset = load_dataset(canonical_name, split="train")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        total_items = len(dataset)
        print(f"[*] Totale campioni da elaborare: {total_items}")

        all_file_chunk_pairs = []

        # Rilevamento automatico delle colonne del codice
        first_row = dataset[0]
        code_field = None
        id_field = None

        for field in ["code", "canonical_solution", "text", "content", "solution"]:
            if field in first_row and isinstance(first_row[field], str):
                code_field = field
                break

        for field in ["task_id", "id", "name", "entry_point"]:
            if field in first_row:
                id_field = field
                break

        for idx, item in enumerate(dataset):
            # Estrazione codice
            if "prompt" in item and "canonical_solution" in item:
                code_text = item["prompt"] + "\n" + item["canonical_solution"]
            elif code_field:
                code_text = item[code_field]
            else:
                code_text = str(list(item.values())[0])

            ident = str(item[id_field]) if id_field else f"sample_{idx:03d}"
            ident_clean = re.sub(r"[^A-Za-z0-9_.-]", "_", ident)
            file_path = f"dataset/{canonical_name.replace('/', '_')}/{ident_clean}.py"

            chunks = self.chunker.chunk_file(file_path, code_text)
            for c in chunks:
                all_file_chunk_pairs.append((file_path, c))

            if len(all_file_chunk_pairs) >= batch_size:
                self.indexer.add_chunks_batch(all_file_chunk_pairs)
                all_file_chunk_pairs = []

        if all_file_chunk_pairs:
            self.indexer.add_chunks_batch(all_file_chunk_pairs)

        self.indexer._resolve_graph_calls()

        stats = {
            "source_type": "huggingface_dataset",
            "source": canonical_name,
            "total_samples": total_items,
            "qdrant_points": self.indexer.client.count(collection_name=self.indexer.collection_name).count,
            "cpg_nodes": self.indexer.graph.number_of_nodes(),
            "cpg_edges": self.indexer.graph.number_of_edges()
        }
        return stats

    def ingest(
        self,
        source: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Metodo universale di routing automatico:
        Dato qualsiasi input (URL GitHub, cartella locale o nome dataset),
        identifica automaticamente la tipologia corretta ed avvia l'ingestione.
        """
        src = source.strip()

        # 1. Riconoscimento GitHub
        if self._is_github_url(src):
            return self.ingest_github_repository(src)

        # 2. Riconoscimento Percorso File System Locale
        if self._is_local_path(src):
            return self.ingest_local_path(src)

        # 3. Riconoscimento Dataset (Hugging Face / Library)
        return self.ingest_dataset_by_name(src, max_samples=max_samples)

    def export_and_view_3d(self, output_html_path: str = "cpg_3d_knowledge_graph.html") -> str:
        """Esporta il grafo di conoscenza corrente in un file HTML 3D navigabile."""
        visualizer = Graph3DVisualizer(self.indexer.graph)
        return visualizer.export_html(output_html_path)

