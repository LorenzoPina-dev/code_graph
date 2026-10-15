from core.chunker import ASTCodeChunker

sample_test_code = '''import os
from math import sqrt

class Calculator:
    """Docstring della classe Calculator"""
    def add(self, a, b):
        """Somma due numeri"""
        return a + b

    def compute(self, val):
        return self.add(val, 10) + sqrt(val)
'''

def test_chunker():
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_file("calculator.py", sample_test_code)
    
    print(f"Total chunks extracted: {len(chunks)}")
    for c in chunks:
        print(f"- [{c.symbol_type.upper()}] {c.symbol_name} (L{c.start_line}-L{c.end_line})")
        print(f"    Docstring: '{c.docstring}'")
        print(f"    Calls: {c.calls}")
        print(f"    Dependencies: {c.dependencies}")

    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    symbols = [c.symbol_name for c in chunks]
    assert "Calculator" in symbols
    assert "Calculator.add" in symbols
    assert "Calculator.compute" in symbols
    
    # Check calls in compute
    compute_chunk = next(c for c in chunks if c.symbol_name == "Calculator.compute")
    assert "add" in compute_chunk.calls or "self.add" in compute_chunk.calls
    assert "sqrt" in compute_chunk.calls
    
    # Check dependencies
    assert "os" in compute_chunk.dependencies
    assert "math.sqrt" in compute_chunk.dependencies
    print("TASK 2 VALIDATION SUCCESSFUL: AST Chunker extracted all symbols, docs, calls, and dependencies.")

if __name__ == "__main__":
    test_chunker()

