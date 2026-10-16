"""
Livello dei FLUSSI del Code Property Graph.

Il chunker produce solo nodi-simbolo e archi `calls` / `contains`. Per capire "cosa succede dopo
il login" servono anche relazioni piu' astratte, che attraversano file, servizi e livelli:

  renders       componente A  -> componente B           (JSX: A mostra B)
  navigates_to  componente/funzione -> schermata/pagina (setScreen("register"), navigate("/x"), <Link>)
  redirects_to  codice -> pagina                        (redirect("/x"), res.redirect, <Navigate>)
  api_call      funzione -> endpoint                    (apiRequest("/identity/me"), fetch, axios)
  proxied_by    endpoint -> gateway                     (tabella di routing / serviceProxy)
  handled_by    endpoint -> funzione che lo implementa  (app.get(...), if (path === "/api/..."))
  publishes     funzione -> evento                      (outbox: "inventory.stock.received.v1")
  consumes      evento   -> funzione                    (worker/servizio che lo gestisce)

Nodi virtuali aggiunti al grafo: `endpoint`, `screen`, `event`, `external` (+ nodi `module` per il
codice a livello di modulo, come le tabelle di routing).

Architettura in due fasi, cosi' la fase 1 e' parallelizzabile e non dipende dal grafo:

  1. FlowFactsExtractor.extract(path, source) -> FileFacts      (per file, Tree-sitter / regex)
  2. FlowLinker(graph, [FileFacts]).link()                      (globale: risolve import, nomi, URL)

Tutte le regole sono euristiche e dichiarative: per supportare un nuovo framework basta aggiungere
un pattern in FlowFactsExtractor (fase 1); il linker non cambia.
"""

import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from tree_sitter import Language, Node, Parser
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

# ----------------------------------------------------------------------------- costanti pubbliche
FLOW_RELATIONS = (
    "renders", "navigates_to", "redirects_to", "api_call",
    "proxied_by", "handled_by", "publishes", "consumes",
)
VIRTUAL_TYPES = ("screen", "endpoint", "event", "external")
JS_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}

HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}
EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+){1,6}\.v\d+$")
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|specs?|e2e|fixtures?)(/|$)|\.(test|spec)\.[a-z0-9]+$|(^|/)test_[^/]*$", re.I
)
_ROUTER_OBJ_RE = re.compile(r"^(app|router|server|fastify|route|routes|\w*Router|\w*App)$")
_CLIENT_CALLEE_RE = re.compile(
    r"^(axios|ky|got|\$fetch|\$http)(\.(get|post|put|patch|delete|head|request))?$"
    r"|^\w*(client|Client|http|Http|api|Api)\w*\.(get|post|put|patch|delete|head)$"
)
_NAV_CALLEE_RE = re.compile(
    r"^(navigate|redirect|router\.(push|replace)|history\.(push|replace)|navigation\.navigate"
    r"|(window\.)?location\.(assign|replace)|(res|response|reply|ctx|c)\.redirect)$"
)
_REDIRECT_CALLEE_RE = re.compile(r"(^|\.)redirect$")
_API_PLACEHOLDER_OK = re.compile(r"API|BACKEND|GATEWAY|SERVER|BASE_URL", re.I)
_API_PLACEHOLDER_NO = re.compile(r"KEYCLOAK|OIDC|AUTH0|ISSUER|IDP|REALM", re.I)


# ----------------------------------------------------------------------------- utilita' sui path
def normalize_path(p: str) -> str:
    """Forma canonica di un path HTTP: parametri -> {}, senza query e senza prefisso /api/vN."""
    p = p.split("?")[0].split("#")[0].strip()
    p = re.sub(r"\$\{[^}]*\}", "{}", p)
    if p.startswith("/"):
        p = re.sub(r":[A-Za-z_][A-Za-z0-9_]*", "{}", p)
    p = re.sub(r"\{[^}]*\}", "{}", p)
    p = re.sub(r"\[[^\]]+\]", "{}", p)
    p = re.sub(r"^/api(?:/v\d+)?(?=/|$)", "", p)
    p = re.sub(r"/{2,}", "/", p)
    p = p.rstrip("/")
    return p or "/"


def paths_match(a: str, b: str) -> bool:
    """Confronto segmento per segmento: {} e * fanno da jolly."""
    sa = [s for s in a.split("/") if s]
    sb = [s for s in b.split("/") if s]
    if len(sa) != len(sb):
        return False
    return all(x == y or x == "{}" or y == "{}" or x == "*" or y == "*" for x, y in zip(sa, sb))


def expand_regex_route(pattern: str, limit: int = 12) -> List[str]:
    """Trasforma una regex di route (es. /^\\/api\\/v1\\/items\\/([0-9a-f-]+)(?:\\/(consume|waste))?$/)
    nei path concreti che descrive: /api/v1/items/{}, /api/v1/items/{}/consume, ..."""
    s = pattern.replace("\\/", "/").replace("\\.", ".")
    s = s.lstrip("^").rstrip("$")
    opt = re.compile(r"\(\?:((?:[^()]|\([^()]*\))*)\)\?")

    def expand(txt: str, depth: int = 0) -> List[str]:
        m = opt.search(txt)
        if not m or depth > 3:
            return [txt]
        before, inner, after = txt[:m.start()], m.group(1), txt[m.end():]
        return expand(before + after, depth + 1) + expand(before + inner + after, depth + 1)

    out: List[str] = []
    for v in expand(s):
        alts = re.search(r"\(([A-Za-z0-9_/\-]+(?:\|[A-Za-z0-9_/\-]+)+)\)", v)
        variants = [v[:alts.start()] + a + v[alts.end():] for a in alts.group(1).split("|")] if alts else [v]
        for w in variants:
            w = re.sub(r"\((?:\?:)?[^()]*\)\??", "{}", w)
            if w.startswith("/") and not re.search(r"[()\\\[\]]", w) and w not in out:
                out.append(w)
    return out[:limit]


def service_of(path: str) -> str:
    """'services/service-identity/src/x.ts' -> 'service-identity'."""
    parts = [p for p in path.split("/") if p]
    if len(parts) > 1 and parts[0] in ("services", "apps", "packages", "microservices", "backend", "py"):
        return parts[1]
    return parts[0] if parts else ""


# ----------------------------------------------------------------------------- fatti per file
@dataclass
class FileFacts:
    path: str
    imports: Dict[str, Tuple[str, str]] = field(default_factory=dict)       # locale -> (modulo, nome|default|*)
    default_export: Optional[str] = None
    consts: Dict[str, List[str]] = field(default_factory=dict)              # NOME -> stringhe
    state_pairs: Dict[str, Tuple[str, int]] = field(default_factory=dict)   # setter -> (variabile, byte)
    local_fns: List[Tuple[str, int, int]] = field(default_factory=list)     # (nome, byte_inizio, byte_fine)
    state_sets: List[Dict[str, Any]] = field(default_factory=list)          # {setter, values, byte, line}
    jsx: List[Dict[str, Any]] = field(default_factory=list)                 # {tag, line, byte, attrs}
    branches: List[Dict[str, Any]] = field(default_factory=list)            # {var, value, tag, line}
    calls: List[Dict[str, Any]] = field(default_factory=list)               # chiamate con primo arg "path"
    wrapper_fns: List[str] = field(default_factory=list)
    routes: List[Dict[str, Any]] = field(default_factory=list)
    gateway: List[Dict[str, Any]] = field(default_factory=list)             # {prefix, base_var, line}
    navs: List[Dict[str, Any]] = field(default_factory=list)                # {line, kind, target}
    events: List[Dict[str, Any]] = field(default_factory=list)              # {line, name, role}


class FlowFactsExtractor:
    """Fase 1: estrae i fatti di un singolo file. Un'istanza per thread (i parser non sono thread-safe)."""

    def __init__(self, include_tests: bool = False):
        self.include_tests = include_tests
        self._parsers: Dict[str, Parser] = {}
        for name, factory in (
            ("javascript", tsjavascript.language),
            ("typescript", tstypescript.language_typescript),
            ("tsx", tstypescript.language_tsx),
        ):
            try:
                self._parsers[name] = Parser(Language(factory()))
            except Exception as e:  # pragma: no cover
                print(f"[!] Parser flussi non disponibile per {name}: {e}")

    def extract(self, path: str, source: str) -> Optional[FileFacts]:
        if not self.include_tests and _TEST_PATH_RE.search(path):
            return None
        ext = PurePosixPath(path).suffix.lower()
        try:
            if ext in JS_EXTS:
                lang = "typescript" if ext == ".ts" else ("tsx" if ext in (".tsx", ".jsx") else "javascript")
                parser = self._parsers.get(lang)
                return _JsWalker(path, source, parser).run() if parser else None
            if ext == ".py":
                return _extract_python(path, source)
        except Exception as e:
            print(f"[!] Estrazione flussi fallita per {path}: {e}")
        return None


# ----------------------------------------------------------------------------- JS / TS / TSX
class _JsWalker:
    def __init__(self, path: str, source: str, parser: Parser):
        self.f = FileFacts(path=path)
        self.src = source.encode("utf8")
        self.lines = source.splitlines()
        self.tree = parser.parse(self.src)
        self.match_bindings: Dict[str, Tuple[str, int, str]] = {}   # nome -> (regex, riga, snippet)
        self.match_used: Set[str] = set()
        self.emitted_routes: Set[Tuple[int, str, str]] = set()

    # ---- helper
    def txt(self, n: Optional[Node]) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf8", errors="replace") if n is not None else ""

    @staticmethod
    def line(n: Node) -> int:
        return n.start_point[0] + 1

    def snippet(self, line: int, span: int = 8) -> str:
        return "\n".join(self.lines[max(0, line - 1): line - 1 + span])[:700]

    def str_value(self, n: Optional[Node]) -> Optional[str]:
        if n is None:
            return None
        if n.type == "string":
            return self.txt(n)[1:-1]
        if n.type == "template_string":
            parts: List[str] = []
            for c in n.children:
                if c.type in ("string_fragment", "escape_sequence"):
                    parts.append(self.txt(c))
                elif c.type == "template_substitution" and c.named_children:
                    parts.append("${" + self.txt(c.named_children[0]) + "}")
            return "".join(parts)
        return None

    def const_strings(self, n: Optional[Node], depth: int = 0) -> List[str]:
        if n is None or depth > 6:
            return []
        s = self.str_value(n)
        if s is not None:
            return [s]
        t = n.type
        if t in ("parenthesized_expression", "as_expression", "satisfies_expression",
                 "non_null_expression", "await_expression"):
            return self.const_strings(n.named_children[0], depth + 1) if n.named_children else []
        if t == "binary_expression":
            op = self.txt(n.child_by_field_name("operator"))
            if op in ("||", "??", "+"):
                return (self.const_strings(n.child_by_field_name("left"), depth + 1)
                        + self.const_strings(n.child_by_field_name("right"), depth + 1))
            return []
        if t == "ternary_expression":
            return (self.const_strings(n.child_by_field_name("consequence"), depth + 1)
                    + self.const_strings(n.child_by_field_name("alternative"), depth + 1))
        if t == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "member_expression":
                return self.const_strings(fn.child_by_field_name("object"), depth + 1)
            return []
        if t == "new_expression":
            args = n.child_by_field_name("arguments")
            return self.const_strings(args.named_children[0], depth + 1) if args is not None and args.named_children else []
        if t == "array":
            out: List[str] = []
            for c in n.named_children:
                out += self.const_strings(c, depth + 1)
            return out
        return []

    def strings_in(self, n: Node) -> List[str]:
        """Stringhe letterali nel sotto-albero, senza scendere nelle funzioni annidate."""
        out, stack = [], [n]
        while stack:
            cur = stack.pop()
            if cur.type in ("arrow_function", "function_expression", "function", "function_declaration"):
                continue
            s = self.str_value(cur) if cur.type == "string" else None
            if s is not None:
                out.append(s)
            stack.extend(reversed(cur.children))
        return out

    def first_component(self, n: Node) -> Optional[str]:
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type in ("jsx_self_closing_element", "jsx_opening_element"):
                name = self.txt(cur.child_by_field_name("name"))
                if name[:1].isupper() and "." not in name:
                    return name
            stack.extend(reversed(cur.children))
        return None

    def eq_comparisons(self, n: Optional[Node]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if n is None:
            return out
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur.type == "binary_expression":
                op = self.txt(cur.child_by_field_name("operator"))
                if op in ("===", "=="):
                    left, right = cur.child_by_field_name("left"), cur.child_by_field_name("right")
                    for a, b in ((left, right), (right, left)):
                        lit = self.str_value(a) if a is not None and a.type == "string" else None
                        if lit is not None and b is not None and b.type in ("identifier", "member_expression"):
                            out.append((self.txt(b)[:40], lit))
            stack.extend(cur.children)
        return out

    # ---- driver
    def run(self) -> FileFacts:
        dispatch = {
            "import_statement": self._import,
            "export_statement": self._export,
            "variable_declarator": self._declarator,
            "function_declaration": self._fn_decl,
            "call_expression": self._call,
            "jsx_self_closing_element": self._jsx,
            "jsx_opening_element": self._jsx,
            "if_statement": self._if,
            "binary_expression": self._binary,
            "ternary_expression": self._ternary,
            "switch_statement": self._switch,
            "assignment_expression": self._assign,
            "string": self._string,
            "object": self._object,
        }
        stack = [self.tree.root_node]
        while stack:
            n = stack.pop()
            handler = dispatch.get(n.type)
            if handler:
                handler(n)
            stack.extend(reversed(n.children))
        # regex di route associate a una variabile ma mai usate in un if: route "ANY"
        for name, (pattern, line, snip) in self.match_bindings.items():
            if name not in self.match_used:
                self._emit_route(line, "ANY", None, None, pattern, snip)
        return self.f

    # ---- import / export
    def _import(self, n: Node) -> None:
        spec = self.str_value(n.child_by_field_name("source"))
        if spec is None:
            return
        for c in n.children:
            if c.type != "import_clause":
                continue
            for cc in c.children:
                if cc.type == "identifier":
                    self.f.imports[self.txt(cc)] = (spec, "default")
                elif cc.type == "named_imports":
                    for sp in cc.children:
                        if sp.type == "import_specifier":
                            nm, al = sp.child_by_field_name("name"), sp.child_by_field_name("alias")
                            imported = self.txt(nm)
                            self.f.imports[self.txt(al) if al is not None else imported] = (spec, imported)
                elif cc.type == "namespace_import":
                    ids = [x for x in cc.children if x.type == "identifier"]
                    if ids:
                        self.f.imports[self.txt(ids[0])] = (spec, "*")

    def _export(self, n: Node) -> None:
        if not self.src[n.start_byte:n.start_byte + 20].decode("utf8", "replace").lstrip().startswith("export default"):
            return
        for c in n.children:
            if c.type in ("function_declaration", "class_declaration", "generator_function_declaration"):
                nm = c.child_by_field_name("name")
                if nm is not None:
                    self.f.default_export = self.txt(nm)
                    return
            if c.type == "identifier":
                self.f.default_export = self.txt(c)
                return
            if c.type == "call_expression":
                args = c.child_by_field_name("arguments")
                for a in (args.named_children if args is not None else []):
                    if a.type == "identifier":
                        self.f.default_export = self.txt(a)
                        return

    # ---- dichiarazioni
    def _declarator(self, n: Node) -> None:
        name, value = n.child_by_field_name("name"), n.child_by_field_name("value")
        if name is None:
            return
        if name.type == "array_pattern" and value is not None and value.type == "call_expression":
            fn = value.child_by_field_name("function")
            if fn is not None and self.txt(fn).split(".")[-1] == "useState":
                ids = [self.txt(c) for c in name.named_children if c.type == "identifier"]
                if len(ids) == 2:
                    self.f.state_pairs[ids[1]] = (ids[0], n.start_byte)
            return
        if name.type != "identifier" or value is None:
            return
        nm = self.txt(name)
        if value.type in ("arrow_function", "function_expression", "function"):
            self.f.local_fns.append((nm, value.start_byte, value.end_byte))
            self._maybe_wrapper(nm, value)
        # variabile legata al risultato di una regex: const m = path.match(/^\/api\/.../)
        vt = self.txt(value)
        if len(vt) < 400:
            m = re.search(r"\.(?:match|exec)\(\s*/((?:\\.|[^/\\\n])+)/[a-z]*\s*\)", vt)
            if m and m.group(1).replace("\\/", "/").lstrip("^").startswith("/api"):
                self.match_bindings[nm] = (m.group(1), self.line(n), self.snippet(self.line(n)))
        parent = n.parent
        if parent is not None and parent.parent is not None and parent.parent.type in ("program", "export_statement"):
            vals = self.const_strings(value)
            if vals:
                self.f.consts[nm] = vals

    def _fn_decl(self, n: Node) -> None:
        nm = n.child_by_field_name("name")
        if nm is None:
            return
        name = self.txt(nm)
        self.f.local_fns.append((name, n.start_byte, n.end_byte))
        self._maybe_wrapper(name, n)

    def _maybe_wrapper(self, name: str, fn: Node) -> None:
        """Una funzione il cui primo parametro finisce dentro fetch()/axios() e' un wrapper HTTP."""
        params = fn.child_by_field_name("parameters")
        first = None
        if params is not None and params.named_children:
            p0 = params.named_children[0]
            ident = p0 if p0.type == "identifier" else next((c for c in p0.children if c.type == "identifier"), None)
            first = self.txt(ident) if ident is not None else None
        else:
            p = fn.child_by_field_name("parameter")
            first = self.txt(p) if p is not None and p.type == "identifier" else None
        body = fn.child_by_field_name("body")
        if not first or body is None:
            return
        stack = [body]
        while stack:
            cur = stack.pop()
            if cur.type == "call_expression":
                callee = self.txt(cur.child_by_field_name("function"))
                if callee == "fetch" or callee.startswith("axios"):
                    args = cur.child_by_field_name("arguments")
                    if args is not None and re.search(rf"\b{re.escape(first)}\b", self.txt(args)):
                        if name not in self.f.wrapper_fns:
                            self.f.wrapper_fns.append(name)
                        return
            stack.extend(cur.children)

    # ---- chiamate
    def _call(self, n: Node) -> None:
        fn, args = n.child_by_field_name("function"), n.child_by_field_name("arguments")
        if fn is None or args is None or fn.end_byte - fn.start_byte > 80:
            return
        callee = self.txt(fn)
        a = args.named_children
        last = callee.split(".")[-1]
        line = self.line(n)

        # setter di stato React: setScreen("register") / setScreen(cond ? "a" : "b")
        if re.fullmatch(r"set[A-Z]\w*", callee) and a:
            vals = [v for arg in a for v in self.strings_in(arg)]
            if vals:
                self.f.state_sets.append({"setter": callee, "values": vals, "byte": n.start_byte, "line": line})

        # navigazione / redirect programmatici
        if _NAV_CALLEE_RE.match(callee) and a:
            target = next((self.str_value(x) for x in a if x.type == "string"), None)
            if target and target.startswith("/") and not target.startswith("//"):
                kind = "redirect" if _REDIRECT_CALLEE_RE.search(callee) else "navigate"
                self.f.navs.append({"line": line, "kind": kind, "target": target})

        if not a:
            return
        first = self.str_value(a[0])
        if first is None:
            return
        pathlike = (first.startswith("/") and not first.startswith("//")) or first.startswith("${") or first.startswith("http")
        if not pathlike:
            return

        method = last.upper() if last.lower() in HTTP_VERBS else None
        for opt in a[1:]:
            if opt.type == "object":
                for pair in opt.named_children:
                    if pair.type == "pair" and self.txt(pair.child_by_field_name("key")).strip("'\"") == "method":
                        mv = self.str_value(pair.child_by_field_name("value"))
                        if mv:
                            method = mv.upper()
        self.f.calls.append({"callee": callee, "line": line, "path": first, "method": method})

        # route in stile Express/Fastify: app.get("/x", handler)
        obj = callee.rsplit(".", 1)[0] if "." in callee else ""
        if first.startswith("/") and (last.lower() in HTTP_VERBS or last in ("all", "use")) and _ROUTER_OBJ_RE.match(obj):
            handler, proxy, upstream = None, False, None
            for arg in a[1:]:
                if arg.type == "call_expression":
                    cfn = self.txt(arg.child_by_field_name("function"))
                    if re.search(r"proxy", cfn, re.I):
                        proxy = True
                        cargs = arg.child_by_field_name("arguments")
                        upstream = self.txt(cargs.named_children[0]) if cargs is not None and cargs.named_children else None
            tail = a[-1] if len(a) > 1 else None
            if tail is not None and tail.type == "identifier":
                handler = self.txt(tail)
            verb = "ALL" if last == "all" else ("USE" if last == "use" else last.upper())
            self._emit_route(line, verb, first, handler, None, self.snippet(line), proxy=proxy, upstream=upstream)

    def _emit_route(self, line: int, method: str, path: Optional[str], handler: Optional[str],
                    pattern: Optional[str], snippet: str, proxy: bool = False,
                    upstream: Optional[str] = None, path_const: Optional[str] = None) -> None:
        key = (line, method, path or pattern or path_const or "")
        if key in self.emitted_routes:
            return
        self.emitted_routes.add(key)
        self.f.routes.append({"line": line, "method": method, "path": path, "pattern": pattern,
                              "path_const": path_const, "handler": handler, "proxy": proxy,
                              "upstream": upstream, "snippet": snippet})

    # ---- route "a mano" (http.createServer): if (method === "GET" && path === "/api/v1/x")
    def _raw_route(self, n: Node, cond: Optional[Node]) -> None:
        if cond is None or cond.end_byte - cond.start_byte > 700:
            return
        ct = self.txt(cond)
        methods = re.findall(r"""\bmethod\s*={2,3}\s*["'](\w+)["']""", ct) or ["ANY"]
        line, snip = self.line(n), self.snippet(self.line(n))
        found = False
        for p in re.findall(r"""\b(?:path|p|pathname|url|route)\s*={2,3}\s*["'](/api[^"']*)["']""", ct):
            found = True
            for m in methods:
                self._emit_route(line, m.upper(), p, None, None, snip)
        consts = re.findall(r"\b(?:path|p|pathname|url)\s*={2,3}\s*([A-Z_][A-Z0-9_]*)\b", ct)
        consts += re.findall(r"\b([A-Z_][A-Z0-9_]*)\.has\(\s*\w+\s*\)", ct)
        for c in consts:
            for m in methods:
                self._emit_route(line, m.upper(), None, None, None, snip, path_const=c)
            found = True
        for rx in re.findall(r"\.match\(\s*/((?:\\.|[^/\\\n])+)/[a-z]*\s*\)", ct):
            if rx.replace("\\/", "/").lstrip("^").startswith("/api"):
                for m in methods:
                    self._emit_route(line, m.upper(), None, None, rx, snip)
                found = True
        for name, (pattern, _, _) in self.match_bindings.items():
            if re.search(rf"\b{re.escape(name)}\b", ct):
                self.match_used.add(name)
                for m in methods:
                    self._emit_route(line, m.upper(), None, None, pattern, snip)
                found = True
        return None if found else None

    # ---- JSX
    def _jsx(self, n: Node) -> None:
        name_n = n.child_by_field_name("name")
        if name_n is None:
            return
        attrs: Dict[str, Dict[str, Any]] = {}
        for c in n.children:
            if c.type != "jsx_attribute":
                continue
            nm, val = None, None
            for cc in c.children:
                if cc.type in ("property_identifier", "jsx_namespace_name") and nm is None:
                    nm = self.txt(cc)
                elif cc.type in ("string", "jsx_expression", "jsx_element", "jsx_self_closing_element", "template_string"):
                    val = cc
            if nm:
                attrs[nm] = self._attr_value(val)
        self.f.jsx.append({"tag": self.txt(name_n), "line": self.line(n), "byte": n.start_byte, "attrs": attrs})

    def _attr_value(self, val: Optional[Node]) -> Dict[str, Any]:
        if val is None:
            return {"kind": "bool"}
        if val.type in ("string", "template_string"):
            return {"kind": "string", "value": self.str_value(val)}
        if val.type == "jsx_expression":
            inner = val.named_children[0] if val.named_children else None
            if inner is None:
                return {"kind": "expr", "text": ""}
            if inner.type == "identifier":
                return {"kind": "ref", "name": self.txt(inner)}
            if inner.type in ("arrow_function", "function_expression", "function"):
                return {"kind": "inline", "start": inner.start_byte, "end": inner.end_byte}
            if inner.type in ("jsx_element", "jsx_self_closing_element"):
                return {"kind": "element", "tag": self.first_component(inner)}
            if inner.type == "string":
                return {"kind": "string", "value": self.str_value(inner)}
            return {"kind": "expr", "text": self.txt(inner)[:80]}
        return {"kind": "element", "tag": self.first_component(val)}

    # ---- schermate: if/&&/?:/switch che scelgono quale componente mostrare
    def _if(self, n: Node) -> None:
        cond, cons = n.child_by_field_name("condition"), n.child_by_field_name("consequence")
        self._raw_route(n, cond)
        comps = self.eq_comparisons(cond)
        if comps and cons is not None:
            tag = self.first_component(cons)
            if tag:
                for var, lit in comps:
                    self.f.branches.append({"var": var, "value": lit, "tag": tag, "line": self.line(n)})

    def _binary(self, n: Node) -> None:
        if self.txt(n.child_by_field_name("operator")) != "&&":
            return
        right = n.child_by_field_name("right")
        if right is None:
            return
        if right.type == "parenthesized_expression" and right.named_children:
            right = right.named_children[0]
        if right.type not in ("jsx_element", "jsx_self_closing_element", "jsx_fragment"):
            return
        tag = self.first_component(right)
        if tag:
            for var, lit in self.eq_comparisons(n.child_by_field_name("left")):
                self.f.branches.append({"var": var, "value": lit, "tag": tag, "line": self.line(n)})

    def _ternary(self, n: Node) -> None:
        cons = n.child_by_field_name("consequence")
        if cons is None:
            return
        tag = self.first_component(cons)
        if tag:
            for var, lit in self.eq_comparisons(n.child_by_field_name("condition")):
                self.f.branches.append({"var": var, "value": lit, "tag": tag, "line": self.line(n)})

    def _switch(self, n: Node) -> None:
        value, body = n.child_by_field_name("value"), n.child_by_field_name("body")
        if value is None or body is None:
            return
        var = self.txt(value).strip("() ")[:40]
        for case in body.children:
            if case.type != "switch_case":
                continue
            lit = self.str_value(case.child_by_field_name("value"))
            tag = self.first_component(case) if lit is not None else None
            if lit is not None and tag:
                self.f.branches.append({"var": var, "value": lit, "tag": tag, "line": self.line(case)})

    # ---- navigazione per assegnamento, eventi, tabella del gateway
    def _assign(self, n: Node) -> None:
        left = self.txt(n.child_by_field_name("left"))
        if left in ("window.location.href", "location.href", "window.location", "location"):
            target = self.str_value(n.child_by_field_name("right"))
            if target and target.startswith("/") and not target.startswith("//"):
                self.f.navs.append({"line": self.line(n), "kind": "navigate", "target": target})

    def _string(self, n: Node) -> None:
        v = self.txt(n)[1:-1]
        if len(v) > 80 or not EVENT_NAME_RE.match(v):
            return
        role = self._event_role(n)
        if role:
            self.f.events.append({"line": self.line(n), "name": v, "role": role})

    def _event_role(self, n: Node) -> Optional[str]:
        p = n.parent
        for _ in range(5):
            if p is None:
                return None
            t = p.type
            if t == "binary_expression":
                return "consume" if self.txt(p.child_by_field_name("operator")) in ("===", "!==", "==", "!=") else None
            if t == "switch_case":
                return "consume"
            if t == "call_expression":
                callee = self.txt(p.child_by_field_name("function"))
                last = callee.split(".")[-1]
                if re.search(r"subscribe|consume|listen|handle|^on$|once|register|includes|^has$", last, re.I):
                    return "consume"
                if re.search(r"publish|emit|enqueue|dispatch|send|produce|event|outbox|record", callee, re.I):
                    return "publish"
                return None
            if t == "variable_declarator":
                name = self.txt(p.child_by_field_name("name"))
                if re.search(r"SUBSCRIB|CONSUM|LISTEN|HANDLED", name, re.I):
                    return "consume"
                if re.search(r"PUBLISH|EMIT|PRODUC", name, re.I):
                    return "publish"
                return None
            if t in ("pair", "object", "array", "arguments", "parenthesized_expression",
                     "as_expression", "new_expression", "satisfies_expression"):
                p = p.parent
                continue
            return None
        return None

    def _object(self, n: Node) -> None:
        keys: Dict[str, Node] = {}
        for pair in n.named_children:
            if pair.type == "pair":
                k = self.txt(pair.child_by_field_name("key")).strip("'\"")
                keys[k] = pair.child_by_field_name("value")
        if "prefix" in keys and "base" in keys:
            prefix = self.str_value(keys["prefix"])
            if prefix and prefix.startswith("/"):
                self.f.gateway.append({"prefix": prefix, "base_var": self.txt(keys["base"]), "line": self.line(n)})


# ----------------------------------------------------------------------------- Python (FastAPI / Flask)
_PY_ROUTE_RE = re.compile(r"""^\s*@(\w+)\.(get|post|put|patch|delete|route)\(\s*['"]([^'"]+)['"](.*)$""")


def _extract_python(path: str, source: str) -> Optional[FileFacts]:
    f = FileFacts(path=path)
    lines = source.splitlines()
    for i, ln in enumerate(lines):
        m = _PY_ROUTE_RE.match(ln)
        if not m:
            continue
        verb, route, rest = m.group(2), m.group(3), m.group(4)
        methods = [verb.upper()]
        if verb == "route":
            mm = re.search(r"methods\s*=\s*\[([^\]]*)\]", rest)
            methods = [x.strip(" '\"").upper() for x in mm.group(1).split(",") if x.strip(" '\"")] if mm else ["GET"]
        handler = None
        for j in range(i + 1, min(i + 8, len(lines))):
            dm = re.match(r"\s*(?:async\s+)?def\s+(\w+)", lines[j])
            if dm:
                handler = dm.group(1)
                break
        for meth in methods:
            f.routes.append({"line": i + 1, "method": meth, "path": route, "pattern": None, "path_const": None,
                             "handler": handler, "proxy": False, "upstream": None,
                             "snippet": "\n".join(lines[i:i + 8])[:700]})
    return f if f.routes else None


# ----------------------------------------------------------------------------- fase 2: collegamento
class FlowLinker:
    """Fase 2: trasforma i fatti dei file in nodi virtuali e archi di flusso nel grafo."""

    def __init__(self, graph: nx.DiGraph, facts: List[Optional[FileFacts]]):
        self.g = graph
        self.facts = [f for f in facts if f is not None]
        self.by_path: Dict[str, FileFacts] = {f.path: f for f in self.facts}
        self.stats: Counter = Counter()
        self.spans: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        self.file_nodes: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.names: Dict[str, List[str]] = defaultdict(list)
        for nid, d in graph.nodes(data=True):
            fp = d.get("file_path")
            if d.get("virtual") or not fp:
                continue
            self.spans[fp].append((d.get("start_line") or 0, d.get("end_line") or 0, nid))
            name = d.get("symbol_name")
            if name:
                self.file_nodes[fp][name] = nid
                self.names[name].append(nid)
        self.known_files: Set[str] = set(self.spans) | set(self.by_path)

    # ---------------------------------------------------------------- utilita' sul grafo
    def _vnode(self, nid: str, typ: str, name: str, file_path: str = "", line: int = 0,
               content: str = "", doc: str = "", **extra: Any) -> str:
        if not self.g.has_node(nid):
            self.g.add_node(nid, file_path=file_path, symbol_name=name, symbol_type=typ,
                            start_line=line, end_line=line, docstring=doc, dependencies=[], calls=[],
                            content=content, virtual=True, **extra)
            self.stats[f"nodes:{typ}"] += 1
        else:
            d = self.g.nodes[nid]
            if content and not d.get("content"):
                d["content"] = content
            if file_path and not d.get("file_path"):
                d["file_path"], d["start_line"], d["end_line"] = file_path, line, line
        return nid

    def _edge(self, u: str, v: str, rel: str, **meta: Any) -> None:
        if u == v or not (self.g.has_node(u) and self.g.has_node(v)):
            return
        if self.g.has_edge(u, v):
            d = self.g[u][v]
            rels = list(d.get("relations") or [d.get("relation")])
            if rel in rels:
                return
            rels.append(rel)
            d["relations"] = rels
            d.setdefault("meta", {})[rel] = meta
        else:
            self.g.add_edge(u, v, relation=rel, relations=[rel], meta={rel: meta})
        self.stats[f"edges:{rel}"] += 1

    def _module_node(self, file: str) -> str:
        nid = f"{file}::<module>"
        stem = PurePosixPath(file).stem
        return self._vnode(nid, "module", f"{stem} (codice di modulo)", file_path=file, line=1,
                           content=f"// Codice a livello di modulo di {file}", doc="Codice fuori da funzioni e classi.")

    def _enclosing(self, file: str, line: int) -> str:
        best, best_span = None, None
        for start, end, nid in self.spans.get(file, []):
            if start <= line <= end and (best_span is None or end - start < best_span):
                best, best_span = nid, end - start
        return best or self._module_node(file)

    def _resolve_module(self, from_file: str, spec: str) -> Optional[str]:
        if spec.startswith("."):
            base = posixpath.normpath(posixpath.join(posixpath.dirname(from_file), spec))
        else:
            s = re.sub(r"^(@/|~/|#/)", "", spec)
            cands = {p for p in self.known_files if p.rsplit(".", 1)[0].endswith("/" + s) or p.rsplit(".", 1)[0] == s}
            return next(iter(cands)) if len(cands) == 1 else None
        for ext in ("", ".tsx", ".ts", ".jsx", ".js", ".mjs", "/index.tsx", "/index.ts", "/index.jsx", "/index.js"):
            if base + ext in self.known_files:
                return base + ext
        return None

    def _resolve_symbol(self, file: str, name: str) -> Optional[str]:
        head = name.split(".")[0]
        here = self.file_nodes.get(file, {})
        if head in here:
            return here[head]
        facts = self.by_path.get(file)
        imp = facts.imports.get(head) if facts else None
        if imp:
            target = self._resolve_module(file, imp[0])
            if target:
                tn = self.file_nodes.get(target, {})
                if imp[1] in ("default", "*"):
                    tf = self.by_path.get(target)
                    for cand in ((tf.default_export if tf else None), head):
                        if cand and cand in tn:
                            return tn[cand]
                    tops = [nid for nm, nid in tn.items()
                            if "." not in nm and self.g.nodes[nid].get("symbol_type") in ("component", "function", "class")]
                    if len(tops) == 1:
                        return tops[0]
                elif imp[1] in tn:
                    return tn[imp[1]]
        ids = self.names.get(head, [])
        return ids[0] if len(ids) == 1 else None

    # ---------------------------------------------------------------- entry point
    def link(self) -> Dict[str, Any]:
        self._link_routes()
        self._link_api_calls()
        self._link_gateway()
        self._link_renders()
        self._link_state_navigation()
        self._link_path_navigation()
        self._link_events()
        rel = {k.split(":", 1)[1]: v for k, v in self.stats.items() if k.startswith("edges:")}
        nodes = {k.split(":", 1)[1]: v for k, v in self.stats.items() if k.startswith("nodes:")}
        return {"flow_edges": sum(rel.values()), "flow_edges_by_relation": rel, "flow_nodes_by_type": nodes}

    # ---------------------------------------------------------------- endpoint
    def _endpoints(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(n, d) for n, d in self.g.nodes(data=True) if d.get("symbol_type") == "endpoint"]

    def _match_endpoints(self, method: str, norm: str) -> List[str]:
        out = []
        for nid, d in self._endpoints():
            if d.get("method") in (method, "ANY") or method == "ANY":
                if paths_match(d.get("path_norm", ""), norm):
                    out.append(nid)
        exact = [n for n in out if self.g.nodes[n].get("method") == method]
        return exact or out

    def _endpoint(self, method: str, norm: str, display: str, file: str = "", line: int = 0, snippet: str = "") -> str:
        nid = f"endpoint::{method} {norm}"
        label = f"{method} {norm}" if method != "ANY" else f"ANY {norm}"
        return self._vnode(nid, "endpoint", label, file_path=file, line=line, content=snippet,
                           doc=f"Endpoint HTTP {label}", method=method, path_norm=norm)

    def _link_routes(self) -> None:
        for f in self.facts:
            for r in f.routes:
                method = r["method"] or "ANY"
                if method == "USE" and r.get("proxy"):
                    continue   # prefisso proxy: gestito in _link_gateway
                paths: List[str] = []
                if r.get("path"):
                    paths.append(r["path"])
                if r.get("path_const"):
                    paths += [v for v in f.consts.get(r["path_const"], []) if v.startswith("/")]
                if r.get("pattern"):
                    paths += expand_regex_route(r["pattern"])
                for p in paths:
                    norm = normalize_path(p)
                    ep = self._endpoint("ANY" if method in ("USE", "ALL") else method, norm, p,
                                        f.path, r["line"], r.get("snippet", ""))
                    if r.get("proxy"):
                        self._edge(ep, self._enclosing(f.path, r["line"]), "proxied_by",
                                   upstream=self._upstream_name(f, r.get("upstream")), line=r["line"])
                    else:
                        handler = self._resolve_symbol(f.path, r["handler"]) if r.get("handler") else None
                        handler = handler or self._enclosing(f.path, r["line"])
                        self._edge(ep, handler, "handled_by", service=service_of(f.path), line=r["line"])

    def _upstream_name(self, f: FileFacts, var: Optional[str]) -> str:
        if not var:
            return ""
        for v in f.consts.get(var.split(".")[-1], []):
            m = re.match(r"https?://([\w.-]+)", v)
            if m:
                return m.group(1)
        return var

    # ---------------------------------------------------------------- chiamate API del client
    def _link_api_calls(self) -> None:
        wrappers = {w for f in self.facts for w in f.wrapper_fns}
        for f in self.facts:
            for c in f.calls:
                callee = c["callee"]
                last = callee.split(".")[-1]
                if not (callee in wrappers or last in wrappers or callee == "fetch" or callee.endswith(".fetch")
                        or _CLIENT_CALLEE_RE.match(callee)):
                    continue
                path, method = c["path"], (c["method"] or "GET")
                caller = self._enclosing(f.path, c["line"])
                internal_path: Optional[str] = None
                external_label: Optional[str] = None
                if path.startswith("/"):
                    internal_path = path
                elif path.startswith("${"):
                    m = re.match(r"\$\{([^}]*)\}(.*)", path)
                    name, rest = (m.group(1), m.group(2)) if m else ("", path)
                    if _API_PLACEHOLDER_OK.search(name) and not _API_PLACEHOLDER_NO.search(name):
                        if re.fullmatch(r"/?(?:\$\{[^}]*\})*", rest):
                            continue   # URL interamente dinamico (es. il wrapper stesso: `${API}${path}`): niente endpoint
                        internal_path = rest if rest.startswith("/") else "/" + rest
                    else:
                        external_label = f"{name}{rest}"
                elif path.startswith("http"):
                    external_label = re.sub(r"^https?://", "", path)
                if internal_path is not None:
                    norm = normalize_path(internal_path)
                    targets = self._match_endpoints(method, norm) or [self._endpoint(method, norm, internal_path)]
                    for ep in targets:
                        self._edge(caller, ep, "api_call", method=method, path=norm, line=c["line"])
                elif external_label:
                    ext = self._vnode(f"external::{method} {external_label}", "external", f"{method} {external_label}",
                                      doc="Servizio esterno (non presente nel repository)", method=method)
                    self._edge(caller, ext, "api_call", method=method, path=external_label, line=c["line"])

    # ---------------------------------------------------------------- gateway
    def _link_gateway(self) -> None:
        entries: List[Tuple[str, str, str, int]] = []   # (prefix_norm, upstream, file, line)
        for f in self.facts:
            for g in f.gateway:
                entries.append((normalize_path(g["prefix"]), self._upstream_name(f, g["base_var"]), f.path, g["line"]))
            for r in f.routes:
                if r["method"] == "USE" and r.get("proxy") and r.get("path"):
                    entries.append((normalize_path(r["path"]), self._upstream_name(f, r.get("upstream")),
                                    f.path, r["line"]))
        for prefix, upstream, file, line in entries:
            gw = self._enclosing(file, line)
            for nid, d in self._endpoints():
                p = d.get("path_norm", "")
                if p == prefix or p.startswith(prefix.rstrip("/") + "/"):
                    self._edge(nid, gw, "proxied_by", upstream=upstream, prefix=prefix, line=line)

    # ---------------------------------------------------------------- UI: componenti renderizzati
    def _link_renders(self) -> None:
        for f in self.facts:
            for j in f.jsx:
                tag = j["tag"]
                if not tag[:1].isupper():
                    continue
                target = self._resolve_symbol(f.path, tag)
                if target:
                    self._edge(self._enclosing(f.path, j["line"]), target, "renders", line=j["line"])

    # ---------------------------------------------------------------- UI: navigazione a stati (screen/tab)
    def _link_state_navigation(self) -> None:
        for f in self.facts:
            if not f.state_pairs or not f.branches:
                continue
            owner_cache: Dict[str, str] = {}

            def related_branches(setter: str, value: str) -> List[Dict[str, Any]]:
                var = (f.state_pairs.get(setter) or ("", 0))[0].lower()
                return [b for b in f.branches if b["value"] == value and var
                        and (b["var"].lower() == var or var in b["var"].lower() or b["var"].lower() in var)]

            def is_screen_state(setter: str) -> bool:
                var = (f.state_pairs.get(setter) or ("", 0))[0].lower()
                return bool(var) and any(var in b["var"].lower() or b["var"].lower() in var for b in f.branches)

            def target_of(setter: str, value: str) -> Optional[str]:
                br = related_branches(setter, value)
                if br:
                    sym = self._resolve_symbol(f.path, br[0]["tag"])
                    if sym:
                        return sym
                # valore senza componente dedicato (es. "app"): schermata virtuale che mostra il componente
                # proprietario dello stato (quello che contiene useState)
                if setter not in owner_cache:
                    byte = f.state_pairs[setter][1]
                    line = _byte_to_line(self.by_path, f.path, byte, f)
                    owner_cache[setter] = self._enclosing(f.path, line)
                owner = owner_cache[setter]
                sid = self._vnode(f"screen::{f.path}::{value}", "screen", f'schermata "{value}"',
                                  file_path=f.path, line=self.g.nodes[owner].get("start_line", 0),
                                  doc=f'Stato "{value}" gestito da {self.g.nodes[owner].get("symbol_name")}')
                self._edge(sid, owner, "renders", via=setter)
                return sid

            def sets_in(start: int, end: int) -> List[Dict[str, Any]]:
                return [s for s in f.state_sets if start <= s["byte"] <= end and s["setter"] in f.state_pairs
                        and is_screen_state(s["setter"])]

            covered: List[Tuple[int, int]] = []
            fn_range = {name: (a, b) for name, a, b in f.local_fns}
            for j in f.jsx:
                tag = j["tag"]
                if not tag[:1].isupper():
                    continue
                child = self._resolve_symbol(f.path, tag)
                if not child:
                    continue
                content = self.g.nodes[child].get("content") or ""
                for prop, av in j["attrs"].items():
                    rng = None
                    if av["kind"] == "inline":
                        rng = (av["start"], av["end"])
                    elif av["kind"] == "ref" and av["name"] in fn_range:
                        rng = fn_range[av["name"]]
                    if rng is None:
                        continue
                    sets = sets_in(*rng)
                    # il figlio deve davvero invocare la prop (dichiarazione + uso => almeno 2 occorrenze)
                    if not sets or len(re.findall(rf"\b{re.escape(prop)}\b", content)) < 2:
                        continue
                    covered.append(rng)
                    for s in sets:
                        for value in s["values"]:
                            tgt = target_of(s["setter"], value)
                            if tgt:
                                self._edge(child, tgt, "navigates_to", via=prop, state=s["setter"], value=value,
                                           line=s["line"])
            # set "diretti": eseguiti dal componente stesso (logout, effetti, guard di sessione)
            for s in f.state_sets:
                if s["setter"] not in f.state_pairs or not is_screen_state(s["setter"]):
                    continue
                if any(a <= s["byte"] <= b for a, b in covered):
                    continue
                src = self._enclosing(f.path, s["line"])
                for value in s["values"]:
                    tgt = target_of(s["setter"], value)
                    if tgt:
                        self._edge(src, tgt, "navigates_to", via=s["setter"], state=s["setter"], value=value,
                                   line=s["line"])

    # ---------------------------------------------------------------- UI: navigazione per URL / router
    def _link_path_navigation(self) -> None:
        ui_routes: Dict[str, str] = {}
        for f in self.facts:
            for j in f.jsx:
                if j["tag"] != "Route":
                    continue
                path = (j["attrs"].get("path") or {}).get("value")
                if not path:
                    continue
                comp = None
                el = j["attrs"].get("element") or {}
                if el.get("kind") == "element":
                    comp = el.get("tag")
                cmp_attr = j["attrs"].get("component") or {}
                if comp is None and cmp_attr.get("kind") == "ref":
                    comp = cmp_attr["name"]
                sym = self._resolve_symbol(f.path, comp) if comp else None
                if sym:
                    ui_routes[normalize_path(path)] = sym

        def go(src: str, target: str, kind: str, line: int, file: str) -> None:
            norm = normalize_path(target)
            dest = next((sym for p, sym in ui_routes.items() if paths_match(p, norm)), None)
            if dest is None:
                dest = self._vnode(f"screen::route::{norm}", "screen", f"pagina {target.split('?')[0]}",
                                   doc="Pagina raggiunta tramite URL")
            self._edge(src, dest, "redirects_to" if kind == "redirect" else "navigates_to", target=target, line=line)

        for f in self.facts:
            for nav in f.navs:
                go(self._enclosing(f.path, nav["line"]), nav["target"], nav["kind"], nav["line"], f.path)
            for j in f.jsx:
                tag, attrs = j["tag"], j["attrs"]
                tgt = None
                kind = "navigate"
                if tag in ("Link", "NavLink"):
                    tgt = (attrs.get("to") or {}).get("value")
                elif tag == "Navigate":
                    tgt, kind = (attrs.get("to") or {}).get("value"), "redirect"
                elif tag == "a":
                    tgt = (attrs.get("href") or {}).get("value")
                if tgt and tgt.startswith("/") and not tgt.startswith("//") and not tgt.startswith("/api"):
                    go(self._enclosing(f.path, j["line"]), tgt, kind, j["line"], f.path)

    # ---------------------------------------------------------------- eventi di dominio
    def _link_events(self) -> None:
        for f in self.facts:
            for ev in f.events:
                node = self._vnode(f"event::{ev['name']}", "event", ev["name"],
                                   doc="Evento di dominio (outbox / bus eventi)")
                owner = self._enclosing(f.path, ev["line"])
                if ev["role"] == "publish":
                    self._edge(owner, node, "publishes", line=ev["line"])
                else:
                    self._edge(node, owner, "consumes", line=ev["line"])


def _byte_to_line(by_path: Dict[str, FileFacts], path: str, byte: int, f: FileFacts) -> int:
    """Stima la riga di un offset usando le chiamate/jsx del file (fallback: prima riga nota)."""
    best = 1
    for coll in (f.jsx, f.state_sets):
        for item in coll:
            if item.get("byte", 1 << 60) <= byte:
                best = max(best, item.get("line", 1))
    return best


def link_flows(graph: nx.DiGraph, facts: List[Optional[FileFacts]]) -> Dict[str, Any]:
    """Scorciatoia: collega i fatti al grafo e restituisce le statistiche."""
    return FlowLinker(graph, facts).link()
