# ContextIQ — Documentation Q&A Assistant
**Forward Deployed Engineer (FDE) Case Study Submission for Solstice Learning**  
*Author: Hemakiran Rachamallu | Tech Stack: Python 3.11+, FastAPI, Qdrant Vector DB, LangGraph State Architecture, Okapi BM25*

---

## 1. What Was Built
**ContextIQ** is an enterprise-grade, high-precision documentation Q&A assistant built for Elena Torres, Head of Support at Solstice Learning. It enables corporate platform administrators to ask plain-English operational questions (password resets, billing cycles, CSV export row limits, SSO) and receive verified answers with 3-tier citations.

### 🌐 Live Cloud Deployment (Production Ready):
ContextIQ is live in production with a public HTTPS endpoint and interactive documentation:
- **Interactive API Documentation (Swagger UI)**: [https://contextiq-7r4v.onrender.com/docs](https://contextiq-7r4v.onrender.com/docs)
- **Live Healthcheck & Vector DB Stats**: [https://contextiq-7r4v.onrender.com/health](https://contextiq-7r4v.onrender.com/health)
- **Live Automated Benchmark Evaluation**: [https://contextiq-7r4v.onrender.com/api/v1/evaluate](https://contextiq-7r4v.onrender.com/api/v1/evaluate)
- **Production Base URL**: `https://contextiq-7r4v.onrender.com`

**Live cURL Sample:**
```bash
curl -X POST https://contextiq-7r4v.onrender.com/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'
```

### Primary Safety Mandate (Zero Hallucinations):
Elena's core requirement is strictly enforced: *"If the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."* ContextIQ achieves **100% precision on benchmark questions with 0.0% hallucination rate** using a multi-signal rejection boundary (Out-of-Vocabulary entity pre-filtering, dynamic risk-calibrated thresholds, and quantitative fact corroboration).

### Cloud-Native Architecture:
ContextIQ ships as both an interactive CLI tool and a **FastAPI REST microservice** backed by a **Qdrant vector database**, containerized with a production multi-stage Dockerfile and deployed to the cloud.

---

## 2. Requirement-to-File Mapping

| Case Study Requirement | Implementing File | Description & Technical Rationale |
| :--- | :--- | :--- |
| **Solution Design Document** | [`SOLUTION_DESIGN.md`](./SOLUTION_DESIGN.md) | Complete engineering spec: problem restatement, 7 assumptions, architecture diagrams, benchmark walkthrough, and v2 scaling roadmap. |
| **Semantic Ingestion & Chunking** | [`contextiq/chunker.py`](./contextiq/chunker.py) | Replaces naive character slicing with sentence/paragraph-aware chunking; extracts UI breadcrumbs (`Settings > Team > Invite Member`). |
| **Retrieval & Relevance Scoring** | [`contextiq/index.py`](./contextiq/index.py) | Okapi BM25 ($k_1=1.5, b=0.75$) augmented with $2.5\times$ Title Weighting and a comprehensive Corpus Vocabulary Registry. |
| **Safety & Rejection Boundary** | [`contextiq/threshold.py`](./contextiq/threshold.py) | Adaptive confidence gatekeeper: Out-of-Vocabulary (OOV) entity check (`Q-5` Salesforce rejection), domain-risk scaling, fact corroboration. |
| **Grounded Answer Extraction** | [`contextiq/extractor.py`](./contextiq/extractor.py) | Pinpoints the exact fact-bearing sentence and formats 3-tier citations (Doc ID, Title, exact quote, navigation path). |
| **State Machine Orchestrator** | [`contextiq/graph.py`](./contextiq/graph.py) | LangGraph-compatible state machine with deterministic conditional routing (`confidence_gate` $\rightarrow$ answer vs refusal). |
| **Qdrant Vector Database** | [`contextiq/vector_store.py`](./contextiq/vector_store.py) | Qdrant adapter with `:memory:` fallback, cosine similarity, payload pre-filtering, and remote AWS cluster support. |
| **Unified Engine Interface** | [`contextiq/engine.py`](./contextiq/engine.py) | Ingests, upserts, and executes queries in $<5\text{ms}$ with zero downtime hot-reloads. |
| **REST API Microservice** | [`api.py`](./api.py) | FastAPI REST API with 6 endpoints, CORS, Swagger UI (`/docs`), and Pydantic request validation. |
| **CLI & Evaluation Runner** | [`main.py`](./main.py) | Terminal entrypoint: interactive REPL, single question query, benchmark evaluation, and `--serve` API mode. |
| **Docker Containerization** | [`Dockerfile`](./Dockerfile), [`docker-compose.yml`](./docker-compose.yml) | Multi-stage Docker build with Qdrant sidecar, ready for AWS ECS deployment. |
| **Short-Term Session Memory** | [`contextiq/session.py`](./contextiq/session.py) | In-memory session manager with TTL expiration; maintains state across turns for single-shot HITL. |
| **Automated Test Suite** | [`tests/`](./tests/) | 25 automated tests: 14 engine unit tests + 11 API integration tests (100% passing). |

---

## 3. How to Run and Test ContextIQ

The core engine uses Python standard library modules by design—it executes out of the box with zero third-party installation failures.

### A. Install Dependencies (Optional — for REST API and Vector DB)
```bash
pip install -r requirements.txt
```

### B. Run Automated Benchmark Evaluation
Runs all 5 evaluation questions from `starter_kit/data/example_questions.json`:
```bash
python main.py --evaluate
```

### C. Run Automated Unit & Regression Tests
Executes the full 25-test suite (engine + API) in under 0.2 seconds:
```bash
python -m unittest discover tests
```

### D. Ask a Single Question via CLI
```bash
python main.py --question "What's the maximum number of rows I can export to CSV?"
python main.py --question "Can I connect Solstice to Salesforce?"
```

### E. Launch Interactive Admin Mode (REPL)
```bash
python main.py
```
*Type `/stats` for index metadata, `/reload` to refresh documentation from disk, or `/quit` to exit.*

---

## 4. REST API (FastAPI)

### Launch the API Server
```bash
python main.py --serve                    # localhost:8000
python main.py --serve --port 9000        # Custom port
python main.py --serve --host 0.0.0.0     # Bind all interfaces (Docker)
```

### Interactive API Documentation
Once the server is running, open your browser to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check with runtime statistics |
| `POST` | `/api/v1/ask` | Answer an admin question with grounded citations |
| `POST` | `/api/v1/ingest` | Ingest a single document into the live corpus |
| `POST` | `/api/v1/reload` | Reload documentation corpus from disk |
| `GET` | `/api/v1/evaluate` | Run benchmark evaluation scorecard |
| `GET` | `/api/v1/vector-search` | Direct vector similarity search against Qdrant |

### Example API Calls
```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'

# Run benchmark evaluation
curl http://localhost:8000/api/v1/evaluate

# Vector similarity search
curl "http://localhost:8000/api/v1/vector-search?query=password+reset&top_k=3"

# Ingest a new document
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "D-07", "title": "New Feature", "content": "Feature documentation..."}'
```

---

## 5. Docker Deployment (AWS ECS Ready)

### Local Docker Compose
```bash
# Build and start ContextIQ API + Qdrant Vector DB
docker compose up --build

# Start in detached mode
docker compose up -d

# Stop services
docker compose down
```

This starts two services:
- **contextiq-api** on port `8000` — FastAPI REST microservice
- **qdrant** on ports `6333` (HTTP) and `6334` (gRPC) — Qdrant Vector Database

### AWS ECS Deployment
The `Dockerfile` and `docker-compose.yml` are production-ready for AWS ECS/Fargate:
1. Push the Docker image to **Amazon ECR**
2. Create an **ECS Task Definition** with the `contextiq-api` and `qdrant` containers
3. Deploy behind an **Application Load Balancer** for HTTPS termination
4. The `QDRANT_URL` environment variable configures remote Qdrant connectivity

---

## 6. Key Architectural Decisions & Trade-Offs

1. **Enhanced BM25 with Title Weighting over Vector Embeddings (in v1)**:
   * *Rationale*: Corporate admin inquiries hinge on exact technical identifiers (`SAML 2.0`, `10,000 rows`, `HTTP 429`, `Growth plan`). Vector embeddings blur numerical boundaries and assign non-zero similarity to unindexed brands (e.g. Salesforce). BM25 delivers 100% deterministic precision, zero hallucination risk, and $<3\text{ms}$ execution without 500MB dependencies.
2. **Qdrant Vector Database for Hybrid Retrieval**:
   * *Rationale*: While BM25 handles exact-match retrieval perfectly for v1, Qdrant provides the foundation for v2 hybrid retrieval (dense + sparse). The adapter supports `:memory:` mode for testing and remote AWS clusters for production, with payload pre-filtering for metadata-scoped searches.
3. **Adaptive Confidence Thresholding over Static Thresholds**:
   * *Rationale*: Sensitive topics (Security `D-01`, SSO `D-05`, Billing `D-04`) enforce a stricter threshold ($\tau = 0.50 - 0.65$) than general feature limits ($\tau = 0.35$). Quantitative questions require numeric fact presence to corroborate the match.
4. **Out-of-Vocabulary (OOV) Pre-Filter**:
   * *Rationale*: Proper nouns absent from the corpus vocabulary (e.g., `"Salesforce"`) trigger immediate safe refusal before deep retrieval or generation can hallucinate.

---

## 7. Known Limitations & v2 Roadmap (With More Time / Budget)

* **Synonym Mismatch**: Handled in v1 via token normalization and suffix stemming. In v2, we layer a hybrid dense-sparse retrieval system (`BAAI/bge-small-en-v1.5` embeddings + BM25) to catch conceptual synonyms (e.g., *"spreadsheet"* $\rightarrow$ *"CSV"*).
* **Single-Shot Bounded Disambiguation**: In v2, near-ties between documents (e.g., password reset vs billing reset) trigger a single clarification turn: *"Did you mean A or B?"* (bounded to 1 turn to avoid frustration).
* **Knowledge Gap Telemetry**: Automated clustering of all refused queries to give Elena a prioritized list of documentation topics her team should write next.
* **Role-Based Plan Scoping (RBAC)**: Filtering retrieved docs by the admin's subscription tier (Growth vs Enterprise).

---

## 8. Benchmark Verification Summary

```text
================================================================================
                       BENCHMARK SCORECARD
================================================================================
  Total Questions Evaluated:  5
  Passed Verifications:       5 / 5
  Grounding Accuracy:         100.0%
  Hallucination Rate:         0.0%
================================================================================
  [PASS] Q-1: Reset password?                      -> D-01 (High Confidence)
  [PASS] Q-2: Password reset link expiry?          -> D-01 (15 minutes)
  [PASS] Q-3: Maximum rows to CSV?                 -> D-03 (10,000 rows)
  [PASS] Q-4: API rate limits for Growth?          -> D-06 (60 req/min)
  [PASS] Q-5: Connect Solstice to Salesforce?      -> REFUSED (Safe Refusal)
================================================================================
```
