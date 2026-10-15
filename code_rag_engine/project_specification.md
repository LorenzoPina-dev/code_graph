# Project Specification & Architectural Blueprint: Enterprise Code RAG Engine

## 1. Obiettivi del Progetto
Realizzare un sistema di **Retrieval-Augmented Generation (RAG) a livello di repository (Repository-Level Code RAG)** orientato alla produzione, capace di indicizzare codebase complesse multi-linguaggio, recuperare il contesto architetturale accurato tramite analisi statica e grafi delle dipendenze, e fornire risposte e generazione di codice prive di allucinazioni tramite una pipeline di valutazione e reranking rigorosa.

---

## 2. Stack Tecnologico e Componenti di Produzione (Zero Mock)

### A. Parsing e Chunking Sintattico (AST)
* **Libreria:** `tree-sitter` e grammatiche specifiche (`tree-sitter-python`, `tree-sitter-typescript`, ecc.).
* **Logica:** Sostituzione totale del chunking a caratteri fissi. Il codice sorgente viene analizzato tramite AST per estrarre nodi logici delimitati (funzioni, metodi, classi e docstring associate), preservando l'integrità sintattica e il contesto semantico locale.

### B. Storage e Ricerca Ibrida (Vector Database)
* **Database:** **Qdrant** (istanza Docker in locale o Qdrant Cloud).
* **Configurazione:**
  * **Vettori Densi (Dense Embeddings):** Generati tramite modelli di embedding specializzati nel codice (es. `text-embedding-3-small` o modelli open-source come `sentence-transformers` focalizzati sul codice da Hugging Face).
  * **Vettori Sparsi / Indici Lexical (Sparse/BM25):** Sfruttati nativamente da Qdrant per garantire la ricerca per parole chiave esatte (nomi di funzioni, variabili, costanti).
  * **Payload strutturato:** Metadati obbligatori per ogni punto: `file_path`, `symbol_name`, `start_line`, `end_line`, `language`, e `dependencies_list`.

### C. Graph RAG e Relazioni Strutturali
* **Libreria / Database a Grafo:** **NetworkX** (per repository di medie dimensioni in memoria) o **Neo4j** (per codebase enterprise).
* **Logica:** Mappatura dei Code Property Graphs (CPG). Ogni nodo rappresenta un simbolo (funzione/classe) e gli archi definiscono le relazioni di chiamata (`calls`), ereditarietà (`inherits`) e importazione (`imports`), permettendo al retriever di navigare nel repository oltre la semplice similarità vettoriale.

### D. Query Rewriting e Reranking
* **Query Expansion:** Utilizzo di un LLM leggero (es. GPT-4o-mini o Llama-3-8B) per applicare tecniche di Multi-Query ed estrazione di termini tecnici prima dell'interrogazione del vector store.
* **Reranking:** Modello **Cross-Encoder** (es. `bge-reranker-large` o Cohere Rerank API) applicato sui top-N candidati (es. top 30) per riordinarli chirurgicamente in base alla rilevanza effettiva prima del passaggio all'LLM generativo.

### E. Framework di Valutazione e Metriche (RAGAS)
* **Libreria:** **RAGAS** (`ragas` Python library) integrata con dataset di benchmark reali (es. *OpenAI HumanEval*, *RepoBench* o subset custom del repository aziendale).
* **Metriche di Produzione da Monitorare:**
  * **Retrieval Metrics:** Hit Rate @ K, MRR (Mean Reciprocal Rank).
  * **Generation Metrics (LLM-as-a-judge):**
    * *Faithfulness:* Misura l'aderenza della risposta al contesto di codice recuperato (prevenzione delle allucinazioni di API/metodi).
    * *Context Precision:* Verifica che il retriever non includa rumore non necessario nei prompt.
    * *Answer Relevancy:* Valuta se la risposta risolve effettivamente il task o la query dell'utente.

---

## 3. Requisiti di Integrazione e Flusso End-to-End

1. **Ingestion Pipeline:** Script di scansione ricorsiva della directory del repository, filtraggio dei file binari/ignorati tramite `.gitignore`, parsing tramite Tree-sitter, generazione degli embedding e upsert in batch su Qdrant con relativi metadati e relazioni a grafo.
2. **Retrieval & Execution Pipeline:** Ricezione della query utente $\rightarrow$ Espansione e riscrittura della query $\rightarrow$ Ricerca ibrida (Dense + Sparse) su Qdrant $\rightarrow$ Interrogazione del grafo per espandere il contesto adiacente $\rightarrow$ Reranking tramite Cross-Encoder $\rightarrow$ Assegnazione del contesto pulito all'LLM generativo.
3. **Evaluation Testing Suite:** Pipeline di unit test automatizzati (integrabile in `pytest` o CI/CD) che esegue periodicamente il dataset di validazione tramite RAGAS per tracciare le metriche di Faithfulness e Context Precision a ogni modifica dell'architettura o dei prompt.