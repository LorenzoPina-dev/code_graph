import pytest
from core.indexer import CodeGraphIndexer
from core.dataset_ingestion import DatasetIngestionPipeline
from core.graph_query import KnowledgeGraphQueryEngine

@pytest.fixture(scope="module")
def populated_query_engine():
    indexer = CodeGraphIndexer(collection_name="test_kg_query_col", location=":memory:")
    pipeline = DatasetIngestionPipeline(indexer=indexer)
    # Ingerisci un subset di 10 problemi HumanEval
    pipeline.ingest_humaneval(split="test", max_samples=10)
    return KnowledgeGraphQueryEngine(indexer=indexer)

def test_summary_statistics(populated_query_engine):
    stats = populated_query_engine.get_summary_statistics()
    assert stats["total_nodes"] >= 10
    assert "function" in stats["counts_by_type"]
    assert stats["counts_by_type"]["function"] >= 10

def test_statistical_question(populated_query_engine):
    res = populated_query_engine.answer_question("Quante funzioni ci sono nel grafo?")
    assert res["type"] == "statistical_answer"
    assert "Totale Componenti Indicizzate" in res["answer"] or "totale di" in res["answer"]
    assert "Function" in res["answer"]

def test_flow_trace_question(populated_query_engine):
    # has_close_elements è il task 0 di HumanEval
    res = populated_query_engine.answer_question("Come viene gestita la funzione has_close_elements?")
    assert res["type"] == "flow_trace_answer"
    assert "has_close_elements" in res["answer"]
    assert "Analisi del Flusso per" in res["answer"]
    assert "code_snippet" in res["flow_details"]

def test_natural_language_retrieval(populated_query_engine):
    res = populated_query_engine.answer_question("Trova la funzione che controlla elementi vicini o soglie")
    assert res["type"] in ["retrieval_answer", "flow_trace_answer"]
    assert len(res["answer"]) > 50

