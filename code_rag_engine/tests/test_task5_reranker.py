from core.reranker import CodeReranker

def test_reranker():
    print("Testing CodeReranker with genuine Cross-Encoder...")
    reranker = CodeReranker()

    query = "Come autenticare e verificare il token dell'utente?"

    candidates = [
        {
            "id": "services/cart.py::checkout",
            "file": "services/cart.py",
            "symbol": "checkout",
            "docstring": "Elabora carrello e pagamento dell'ordine.",
            "content": "def checkout(cart_items):\n    return sum(item['price'] for item in cart_items)",
            "score": 0.5
        },
        {
            "id": "services/auth.py::authenticate",
            "file": "services/auth.py",
            "symbol": "authenticate",
            "docstring": "Verifica la validità del token utente.",
            "content": "def authenticate(username, token):\n    return verify_token_db(token)",
            "score": 0.4
        }
    ]

    reranked = reranker.rerank(query, candidates, top_n=2)
    print("Reranked results:")
    for r in reranked:
        print(f"- {r['symbol']} (Rerank score: {r['rerank_score']:.4f})")

    assert len(reranked) == 2
    # Il cross-encoder deve classificare 'authenticate' al primo posto rispetto a 'checkout'
    assert reranked[0]["symbol"] == "authenticate", f"Expected 'authenticate' top 1, got {reranked[0]['symbol']}"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    print("TASK 5 VALIDATION SUCCESSFUL: Cross-Encoder accurately scored and reranked candidates!")

if __name__ == "__main__":
    test_reranker()

