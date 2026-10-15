from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from tree_sitter import Language, Parser, Node

# Import dei moduli grammatica di Tree-sitter installati
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_c_sharp as tscsharp
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust

@dataclass
class CodeChunk:
    file_path: str
    symbol_name: str
    symbol_type: str  # "function", "class", "method", "module", "struct", "interface"
    start_line: int
    end_line: int
    code_content: str
    docstring: str = ""
    language: str = "python"
    dependencies: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    parent_symbol: Optional[str] = None

class ASTCodeChunker:
    """
    Parser sintattico multilingua orientato a repository reali.
    Supporta:
    - Python (.py)
    - C e C++ (.c, .h, .cpp, .hpp, .cc, .cxx, .ino)
    - JavaScript e TypeScript (.js, .jsx, .ts, .tsx, .mjs, .cjs)
    - Java (.java)
    - C# (.cs)
    - Go (.go)
    - Rust (.rs)
    - Fallback regex / boundary per altri linguaggi (es. R, shell, ecc.)
    """

    EXT_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".c": "c",
        ".h": "cpp",       # Spesso header C++ o C
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".ino": "cpp",
        ".cs": "c_sharp",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
    }

    def __init__(self):
        self._parsers: Dict[str, Parser] = {}
        self._init_parsers()

    def _init_parsers(self):
        """Inizializza i parser Tree-sitter per ciascun linguaggio supportato."""
        lang_factories = {
            "python": tspython.language,
            "javascript": tsjavascript.language,
            "typescript": tstypescript.language_typescript,
            "tsx": tstypescript.language_tsx,
            "c": tsc.language,
            "cpp": tscpp.language,
            "c_sharp": tscsharp.language,
            "java": tsjava.language,
            "go": tsgo.language,
            "rust": tsrust.language,
        }

        for lang_name, factory in lang_factories.items():
            try:
                lang_obj = Language(factory())
                self._parsers[lang_name] = Parser(lang_obj)
            except Exception as e:
                print(f"[!] Impossibile inizializzare parser per {lang_name}: {e}")

    def detect_language(self, file_path: str) -> str:
        """Rileva il linguaggio di programmazione dall'estensione del file."""
        ext = Path(file_path).suffix.lower()
        return self.EXT_TO_LANG.get(ext, "unknown")

    def _extract_docstring(self, node: Node, source_bytes: bytes, language: str) -> str:
        """Estrae la docstring o i commenti di intestazione associati."""
        # Controlla se c'è un commento immediatamente precedente (valido per C/C++, Java, Rust, JS, ecc.)
        prev = node.prev_sibling
        if prev and prev.type in ["comment", "block_comment", "line_comment"]:
            comm = source_bytes[prev.start_byte:prev.end_byte].decode("utf8", errors="replace")
            return comm.strip("/*# \n\t")

        # Per Python, controlla l'espressione stringa all'inizio del body
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "expression_statement":
                    for sub in child.children:
                        if sub.type == "string":
                            doc = source_bytes[sub.start_byte:sub.end_byte].decode("utf8", errors="replace")
                            return doc.strip('"""\'\'\' \n\t')
                elif child.type not in ["comment"]:
                    break
        return ""

    def _extract_calls(self, node: Node, source_bytes: bytes, language: str) -> List[str]:
        """Estrae le chiamate di funzione/metodo rilevate nell'AST indipendentemente dal linguaggio."""
        calls = set()
        call_types = {
            "call", "call_expression", "method_invocation", "invocation_expression"
        }

        def scan_calls(curr: Node):
            if curr.type in call_types:
                func_node = curr.child_by_field_name("function") or curr.child_by_field_name("name")
                if func_node:
                    call_text = source_bytes[func_node.start_byte:func_node.end_byte].decode("utf8", errors="replace")
                    # Gestione chiamate metodo (obj.foo() o obj->foo() o Foo::bar())
                    for sep in [".", "->", "::"]:
                        if sep in call_text:
                            calls.add(call_text.split(sep)[-1])
                    calls.add(call_text)
            for child in curr.children:
                if child.type not in ["function_definition", "class_definition", "method_definition", "class_declaration"]:
                    scan_calls(child)

        scan_calls(node)
        return sorted(list(calls))

    def _extract_dependencies(self, root_node: Node, source_bytes: bytes, language: str) -> List[str]:
        """Rileva le dipendenze importate nel file (#include, import, using, use, require)."""
        deps = []
        for child in root_node.children:
            # C/C++: preproc_include (#include <foo.h> o "foo.h")
            if child.type == "preproc_include":
                path_node = child.child_by_field_name("path")
                if path_node:
                    raw = source_bytes[path_node.start_byte:path_node.end_byte].decode("utf8", errors="replace")
                    deps.append(raw.strip('<>"'))

            # Python: import_statement / import_from_statement
            elif child.type == "import_statement":
                for name_node in child.children_by_field_name("name"):
                    raw = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf8", errors="replace")
                    deps.append(raw.strip())
            elif child.type == "import_from_statement":
                module_node = child.child_by_field_name("module_name")
                module_name = source_bytes[module_node.start_byte:module_node.end_byte].decode("utf8", errors="replace") if module_node else ""
                for name_node in child.children_by_field_name("name"):
                    sub_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf8", errors="replace")
                    deps.append(f"{module_name}.{sub_name}" if module_name else sub_name)

            # JS/TS: import_statement / export_statement
            elif child.type == "import_statement":
                source_node = child.child_by_field_name("source")
                if source_node:
                    raw = source_bytes[source_node.start_byte:source_node.end_byte].decode("utf8", errors="replace")
                    deps.append(raw.strip('\'"'))

            # Java: import_declaration
            elif child.type == "import_declaration":
                raw = source_bytes[child.start_byte:child.end_byte].decode("utf8", errors="replace")
                deps.append(raw.replace("import", "").strip("; \n\t"))

            # C#: using_directive
            elif child.type == "using_directive":
                raw = source_bytes[child.start_byte:child.end_byte].decode("utf8", errors="replace")
                deps.append(raw.replace("using", "").strip("; \n\t"))

            # Rust: use_declaration
            elif child.type == "use_declaration":
                raw = source_bytes[child.start_byte:child.end_byte].decode("utf8", errors="replace")
                deps.append(raw.replace("use", "").strip("; \n\t"))

            # Go: import_declaration
            elif child.type == "import_declaration":
                raw = source_bytes[child.start_byte:child.end_byte].decode("utf8", errors="replace")
                deps.append(raw.replace("import", "").strip("() \n\t\"'"))

        return deps

    def chunk_file(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """
        Esegue il parsing sintattico del sorgente in qualunque linguaggio supportato,
        estraendo simboli, docstring, chiamate e dipendenze.
        """
        language = self.detect_language(file_path)
        parser = self._parsers.get(language)

        lines = source_code.splitlines()
        chunks: List[CodeChunk] = []

        # Se il linguaggio non ha un parser Tree-sitter (es. R o estensione non registrata),
        # utilizziamo il fallback a livello di modulo o segmentazione logica
        if not parser:
            if source_code.strip():
                chunks.append(CodeChunk(
                    file_path=file_path,
                    symbol_name=Path(file_path).stem,
                    symbol_type="module",
                    start_line=1,
                    end_line=len(lines),
                    code_content=source_code,
                    language=language
                ))
            return chunks

        source_bytes = bytes(source_code, "utf8")
        try:
            tree = parser.parse(source_bytes)
            root_node = tree.root_node
        except Exception:
            return [CodeChunk(
                file_path=file_path,
                symbol_name=Path(file_path).stem,
                symbol_type="module",
                start_line=1,
                end_line=len(lines),
                code_content=source_code,
                language=language
            )]

        file_deps = self._extract_dependencies(root_node, source_bytes, language)

        # Mappatura dei tipi di nodi Tree-sitter per i diversi linguaggi
        class_types = {
            "class_definition", "class_declaration", "class_specifier", 
            "struct_specifier", "interface_declaration", "enum_declaration",
            "impl_item", "struct_item", "type_declaration"
        }
        func_types = {
            "function_definition", "function_declaration", "function_item",
            "method_definition", "method_declaration", "constructor_declaration",
            "arrow_function"
        }

        def get_node_name(n: Node) -> str:
            name_n = (
                n.child_by_field_name("name") or 
                n.child_by_field_name("declarator") or 
                n.child_by_field_name("identifier")
            )
            if name_n:
                # In C/C++, il declarator può essere annidato (es. function_declarator)
                if name_n.type == "function_declarator":
                    sub = name_n.child_by_field_name("declarator")
                    if sub:
                        return source_bytes[sub.start_byte:sub.end_byte].decode("utf8", errors="replace")
                return source_bytes[name_n.start_byte:name_n.end_byte].decode("utf8", errors="replace")
            return ""

        def traverse(node: Node, parent_class: Optional[str] = None):
            ntype = node.type

            # CLASSI / STRUTTURE / INTERFACCE
            if ntype in class_types:
                raw_name = get_node_name(node) or "AnonymousClass"
                # Riconosci se è struct o interface
                sym_type = "struct" if "struct" in ntype else ("interface" if "interface" in ntype else "class")
                start_row, _ = node.start_point
                end_row, _ = node.end_point
                content = "\n".join(lines[start_row:end_row + 1])
                doc = self._extract_docstring(node, source_bytes, language)
                calls = self._extract_calls(node, source_bytes, language)

                chunks.append(CodeChunk(
                    file_path=file_path,
                    symbol_name=raw_name,
                    symbol_type=sym_type,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    code_content=content,
                    docstring=doc,
                    language=language,
                    dependencies=file_deps,
                    calls=calls,
                    parent_symbol=parent_class
                ))

                body = (
                    node.child_by_field_name("body") or 
                    node.child_by_field_name("declaration_list") or
                    node.child_by_field_name("field_declaration_list")
                )
                if body:
                    for child in body.children:
                        traverse(child, parent_class=raw_name)
                else:
                    for child in node.children:
                        traverse(child, parent_class=raw_name)
                return

            # FUNZIONI / METODI
            elif ntype in func_types:
                raw_name = get_node_name(node) or "anonymous"
                # Pulizia sintattica del nome in C/C++ se include parametri
                if "(" in raw_name:
                    raw_name = raw_name.split("(")[0].strip()

                sym_type = "method" if parent_class else "function"
                full_name = f"{parent_class}::{raw_name}" if (parent_class and language in ["cpp", "c", "rust"]) else (
                    f"{parent_class}.{raw_name}" if parent_class else raw_name
                )
                start_row, _ = node.start_point
                end_row, _ = node.end_point
                content = "\n".join(lines[start_row:end_row + 1])
                doc = self._extract_docstring(node, source_bytes, language)
                calls = self._extract_calls(node, source_bytes, language)

                chunks.append(CodeChunk(
                    file_path=file_path,
                    symbol_name=full_name,
                    symbol_type=sym_type,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    code_content=content,
                    docstring=doc,
                    language=language,
                    dependencies=file_deps,
                    calls=calls,
                    parent_symbol=parent_class
                ))
                return

            for child in node.children:
                traverse(child, parent_class)

        traverse(root_node)

        # Fallback a livello globale di modulo se non ci sono definizioni esplicite
        if not chunks and source_code.strip():
            chunks.append(CodeChunk(
                file_path=file_path,
                symbol_name=Path(file_path).stem,
                symbol_type="module",
                start_line=1,
                end_line=len(lines),
                code_content=source_code,
                language=language,
                dependencies=file_deps,
                calls=self._extract_calls(root_node, source_bytes, language)
            ))

        return chunks