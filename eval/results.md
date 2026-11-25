Questions: 30, documents: 4, top_k: 5

| Pipeline | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| Fixed-size chunks, vector search (baseline) | 83% | 100% | 0.91 |
| Structure-aware chunks, vector search | 93% | 100% | 0.96 |
| Structure-aware chunks, hybrid search (RRF) | 83% | 100% | 0.91 |
| Structure-aware chunks, hybrid + reranker | 97% | 100% | 0.98 |
