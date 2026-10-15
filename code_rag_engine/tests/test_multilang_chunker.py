import pytest
from core.chunker import ASTCodeChunker

def test_multilang_parsing():
    chunker = ASTCodeChunker()

    # 1. C++
    cpp_code = """
    #include <iostream>
    #include "radio.h"

    class MeshRouter {
    public:
        void routePacket(int packetId) {
            sendRadio(packetId);
        }
        void sendRadio(int id) {}
    };

    int main() {
        MeshRouter router;
        router.routePacket(42);
        return 0;
    }
    """
    cpp_chunks = chunker.chunk_file("src/router.cpp", cpp_code)
    symbols = [c.symbol_name for c in cpp_chunks]
    print("C++ extracted:", symbols)
    assert any("MeshRouter" in s for s in symbols)
    assert any("routePacket" in s for s in symbols)
    assert any("main" in s for s in symbols)

    # 2. JavaScript / Node.js
    js_code = """
    import express from 'express';
    const app = express();

    class AuthService {
        verifyToken(token) {
            return checkDb(token);
        }
    }

    function startServer(port) {
        app.listen(port);
    }
    """
    js_chunks = chunker.chunk_file("server.js", js_code)
    js_symbols = [c.symbol_name for c in js_chunks]
    print("JS extracted:", js_symbols)
    assert any("AuthService" in s for s in js_symbols)
    assert any("verifyToken" in s for s in js_symbols)
    assert any("startServer" in s for s in js_symbols)

    # 3. Java
    java_code = """
    package com.example;
    import java.util.List;

    public class PaymentGateway {
        public boolean processPayment(double amount) {
            return validateAmount(amount);
        }
        private boolean validateAmount(double a) { return a > 0; }
    }
    """
    java_chunks = chunker.chunk_file("PaymentGateway.java", java_code)
    java_symbols = [c.symbol_name for c in java_chunks]
    print("Java extracted:", java_symbols)
    assert any("PaymentGateway" in s for s in java_symbols)
    assert any("processPayment" in s for s in java_symbols)

    # 4. Go
    go_code = """
    package main
    import "fmt"

    func CalculateHash(data string) string {
        return fmt.Sprintf("%x", data)
    }
    """
    go_chunks = chunker.chunk_file("hash.go", go_code)
    go_symbols = [c.symbol_name for c in go_chunks]
    print("Go extracted:", go_symbols)
    assert any("CalculateHash" in s for s in go_symbols)

    # 5. Rust
    rust_code = """
    struct NodeConfig {
        id: u32,
    }

    fn init_node(id: u32) -> NodeConfig {
        NodeConfig { id }
    }
    """
    rust_chunks = chunker.chunk_file("node.rs", rust_code)
    rust_symbols = [c.symbol_name for c in rust_chunks]
    print("Rust extracted:", rust_symbols)
    assert any("NodeConfig" in s for s in rust_symbols)
    assert any("init_node" in s for s in rust_symbols)

    # 6. C#
    cs_code = """
    using System;

    namespace App {
        public class Worker {
            public void DoWork() {
                Console.WriteLine("working");
            }
        }
    }
    """
    cs_chunks = chunker.chunk_file("Worker.cs", cs_code)
    cs_symbols = [c.symbol_name for c in cs_chunks]
    print("C# extracted:", cs_symbols)
    assert any("Worker" in s for s in cs_symbols)
    assert any("DoWork" in s for s in cs_symbols)

    print("ALL MULTI-LANGUAGE GRAMMARS VALIDATED SUCCESSFULLY!")

if __name__ == "__main__":
    test_multilang_parsing()

