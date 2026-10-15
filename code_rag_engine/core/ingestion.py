import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathspec
from .chunker import ASTCodeChunker, CodeChunk
from .indexer import CodeGraphIndexer

SUPPORTED_EXTENSIONS = {
    ".py",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".ino",  # C / C++ (Arduino, Embedded, Firmware)
    ".js", ".jsx", ".mjs", ".cjs",                     # JavaScript / Node.js
    ".ts", ".tsx",                                      # TypeScript
    ".java",                                            # Java
    ".cs",                                              # C# (.NET)
    ".go",                                              # Go
    ".rs",                                              # Rust
    ".r", ".R",                                         # R
    ".php", ".rb", ".swift", ".kt"                      # Fallback / altri linguaggi
}

# Dimensione massima file da processare (evita file generati / minified enormi)
MAX_FILE_SIZE_BYTES = 500_000   # 500 KB

# Numero di worker per il parsing parallelo dei file
_CPU_WORKERS = min(8, (os.cpu_count() or 4))


class RepositoryIngestionPipeline:
    """
    Pipeline di scansione, filtraggio e indicizzazione ricorsiva di repository completi.

    Ottimizzazioni rispetto alla versione precedente:
    - Parsing dei file in parallelo (ThreadPoolExecutor, N worker = CPU count)
    - Skip automatico dei file già processati tramite SHA-256 content hash
    - _resolve_graph_calls() eseguita una sola volta a fine ingestione (non ad ogni batch)
    - File troppo grandi (> 500 KB) saltati automaticamente (tipicamente generati / minified)
    - Batch di upsert Qdrant ottimizzato (dimensione 200)
    """

    def __init__(self, indexer: CodeGraphIndexer, chunker: Optional[ASTCodeChunker] = None):
        self.indexer = indexer
        self.chunker = chunker or ASTCodeChunker()
        self._seen_hashes: set = set()

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_gitignore_spec(self, repo_dir: Path) -> Optional[pathspec.PathSpec]:
        """Carica le regole .gitignore se presenti nella root del repository."""
        gitignore_path = repo_dir / ".gitignore"
        patterns = [
            ".git", ".git/*", "__pycache__", "__pycache__/*", "*.pyc",
            ".pytest_cache", ".venv", "venv", "node_modules", "dist", "build",
            "*.egg-info", "*.so", "*.dll", "*.exe", "*.bin",
            "*.min.js", "*.min.css", "*.map",               # Minified / source-maps
            "*.lock", "*.sum", "package-lock.json",         # Lock files
        ]
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                    patterns.extend(f.read().splitlines())
            except Exception:
                pass
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    @staticmethod
    def _file_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def _parse_file(self, file_path: Path, rel_path: str) -> tuple:
        """
        Worker eseguito in parallelo: legge e chunka un singolo file.
        Restituisce (rel_path, chunks) oppure (rel_path, None) in caso di errore.
        """
        try:
            size = file_path.stat().st_size
            if size == 0 or size > MAX_FILE_SIZE_BYTES:
                return (rel_path, None)

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()

            # Deduplicazione tramite hash del contenuto
            fhash = self._file_hash(source_code)
            if fhash in self._seen_hashes:
                return (rel_path, None)
            self._seen_hashes.add(fhash)

            chunks = self.chunker.chunk_file(rel_path, source_code)
            return (rel_path, chunks if chunks else None)
        except Exception as e:
            return (rel_path, None)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def scan_and_index_repository(
        self,
        repo_path: str,
        batch_size: int = 200,          # ↑ da 50 a 200
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Scansiona ricorsivamente la directory indicando i file validi e indicizzandoli.
        Il parsing dei file avviene in parallelo; l'upsert Qdrant è sequenziale in batch.
        """
        root = Path(repo_path).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Directory repository non valida: {repo_path}")

        spec = self._load_gitignore_spec(root)

        # ── Fase 1: colleziona tutti i file validi ──────────────────────
        valid_files: List[tuple] = []  # (file_path, rel_path)
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(root).as_posix()
            if spec and spec.match_file(rel_path):
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            valid_files.append((file_path, rel_path))

        total_files = len(valid_files)
        if show_progress:
            print(f"[*] File candidati trovati: {total_files} | Worker paralleli: {_CPU_WORKERS}")

        # ── Fase 2: parsing parallelo dei file ─────────────────────────
        # Un chunker per worker (i parser Tree-sitter non sono thread-safe se condivisi)
        indexed_files = []
        total_chunks = 0
        pending_pairs: List[tuple] = []   # (rel_path, chunk)

        def make_chunker():
            return ASTCodeChunker()

        with ThreadPoolExecutor(max_workers=_CPU_WORKERS,
                                initializer=None) as executor:
            # Creiamo un chunker fresco per ogni worker usando thread-local
            import threading
            _local = threading.local()

            def _worker(args):
                fp, rp = args
                if not hasattr(_local, "chunker"):
                    _local.chunker = ASTCodeChunker()
                try:
                    size = fp.stat().st_size
                    if size == 0 or size > MAX_FILE_SIZE_BYTES:
                        return (rp, None)
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        source_code = f.read()
                    fhash = self._file_hash(source_code)
                    if fhash in self._seen_hashes:
                        return (rp, None)
                    self._seen_hashes.add(fhash)
                    chunks = _local.chunker.chunk_file(rp, source_code)
                    return (rp, chunks if chunks else None)
                except Exception:
                    return (rp, None)

            futures = {executor.submit(_worker, args): args for args in valid_files}
            done = 0
            for future in as_completed(futures):
                done += 1
                rel_path, chunks = future.result()
                if chunks:
                    indexed_files.append(rel_path)
                    total_chunks += len(chunks)
                    for c in chunks:
                        pending_pairs.append((rel_path, c))

                # Flush batch quando raggiunge la dimensione target
                if len(pending_pairs) >= batch_size:
                    self.indexer.add_chunks_batch(pending_pairs, resolve_calls=False)
                    pending_pairs = []

                if show_progress and done % 200 == 0:
                    print(f"  -> Progresso: {done}/{total_files} file elaborati...")

        # ── Fase 3: flush del batch residuo ────────────────────────────
        if pending_pairs:
            self.indexer.add_chunks_batch(pending_pairs, resolve_calls=False)

        # ── Fase 4: resolve_calls UNA SOLA VOLTA a fine ingestione ─────
        if show_progress:
            print("[*] Costruzione archi 'calls' nel grafo CPG...")
        self.indexer._resolve_graph_calls()

        return {
            "indexed_files_count": len(indexed_files),
            "total_chunks_count": total_chunks,
            "indexed_files": indexed_files,
            "cpg_nodes": self.indexer.graph.number_of_nodes(),
            "cpg_edges": self.indexer.graph.number_of_edges()
        }
