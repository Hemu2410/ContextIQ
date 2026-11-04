# ContextIQ: Solution Design & Architecture Specification
**Case Study 5: Documentation Q&A Assistant for Solstice Learning**  
*Author: Engineering Candidate*  
*Target Reviewer: AI & Technical Review Panel / Elena Torres (Head of Support)*

---

## 1. Problem Restatement & Core Objective

Solstice Learning provides a corporate training platform. Support leadership (Elena Torres) observes that corporate administrators frequently ask repetitive questions regarding standard platform operations (password resets, billing cycles, CSV export row limits, SSO configuration). While these answers exist in the product documentation, administrators struggle with manual search friction.

### Primary Constraint:
> *"The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."* — Elena Torres

**The core objective of ContextIQ** is to deliver a high-precision, low-latency, zero-hallucination assistant that provides direct answers backed by verifiable documentation citations when confident, and deterministically refuses to answer when documentation is absent or insufficient.

---

## 2. Core Technical & Product Assumptions

To resolve the ambiguities in Elena's requirements, the following seven technical and operational assumptions govern the design of ContextIQ:

### Assumption 1: Asymmetric Error Cost (Precision $\gg$ Recall)
* **Rationale**: In enterprise SaaS administration, a false positive (a hallucinated or inaccurate answer) carries severe business consequences—e.g., security misconfigurations, unauthorized session management, or incorrect billing expectations. Conversely, a false negative (saying *"I don't have an answer for that"*) simply maintains the status quo by routing the admin to human support.
* **Design Decision**: The system is intentionally biased toward precision. When retrieval scores or evidence signals fall into ambiguous ranges, the system must prioritize safe refusal over speculative answering.

### Assumption 2: Adaptive, Risk-Calibrated Confidence Thresholding
* **Rationale**: A uniform, static confidence threshold across all inquiries is fundamentally flawed. Inquiries regarding security configurations (`D-01` Passwords, `D-05` SAML SSO) or financial contracts (`D-04` Billing & Refunds) carry higher operational risk than inquiries about general UI utilities (`D-03` CSV report downloads).
* **Design Decision**: Threshold $\tau$ is dynamic:
  * **High-Risk Domain Threshold ($\tau = 0.65$)**: Applied to security, authentication, and billing operations.
  * **Standard Informational Threshold ($\tau = 0.40$)**: Applied to feature limits and navigation.
  * **Fact-Type Corroboration**: If a query asks a quantitative question (*"How long..."*, *"What is the maximum limit..."*), the threshold dynamically relaxes if and only if the candidate passage explicitly contains the corresponding numerical or temporal fact type (e.g., *"15 minutes"*, *"10,000 rows"*). If no matching fact type exists, the threshold escalates to trigger rejection.

### Assumption 3: Out-of-Vocabulary (OOV) Entity Pre-Filter
* **Rationale**: Administrators will ask about third-party tools, vendor platforms, or features not supported by Solstice (e.g., Benchmark Query `Q-5`: *"Can I connect Solstice to Salesforce?"*). When a query focuses on a named entity or proper noun completely absent from the entire documentation corpus, any retrieval match will be purely coincidental noise.
* **Design Decision**: An entity pre-filtering layer extracts proper nouns and key technical terms from the query. If a core entity has zero occurrences across the document vocabulary, ContextIQ immediately triggers the standardized refusal response without proceeding to deeper retrieval or generation.

### Assumption 4: Semantic Document Boundaries over Blind Character Slicing
* **Rationale**: The baseline starter kit (`docs_qa_utils.py`) employs arbitrary 400-character slicing (`text[start:start+400]`), which cuts words and logical clauses in half (e.g., splitting between *"expire after"* and *"15 minutes"*).
* **Design Decision**: Documents are chunked by natural semantic boundaries (paragraphs, sentences, or structured article sections). Chunks must preserve complete grammatical propositions so that facts remain intact.

### Assumption 5: Verifiable Extractive Grounding & Citation Mandate
* **Rationale**: Enterprise administrators will not trust an opaque, conversational black box. Every accepted response must have 100% auditable provenance.
* **Design Decision**: Every answer produced by ContextIQ must explicitly return:
  1. The direct factual answer statement.
  2. The source document identifier (`doc_id`).
  3. The document `title`.
  4. The verbatim supporting quote extracted directly from the text.

### Assumption 6: Low-Latency, Zero-Cold-Start In-Memory Execution
* **Rationale**: The seed dataset consists of 6 core articles, and typical support documentation centers scale from tens to thousands of articles. Introducing heavy distributed vector databases, external GPU services, or mandatory external API keys introduces network failure modes, cold-start latency, and running costs.
* **Design Decision**: ContextIQ's core engine is architected as an in-memory, deterministic retrieval and extraction pipeline (enhanced BM25 with title-weighting and lexical-semantic scoring). It executes in $<5\text{ms}$ with zero external service dependencies, ensuring 100% test reproducibility across environments.

### Assumption 7: Graceful Escalation (No Dead-End UX)
* **Rationale**: When the system correctly refuses to answer (*"I don't have an answer for that in the documentation"*), terminating the session abruptly frustrates the administrator and leaves their operational blocker unresolved.
* **Design Decision**: Every refusal is paired with a clear handoff path:
  > *"I don't have an answer for that based on our documentation. Would you like to submit a ticket to Elena's support team at support@solsticelearning.com?"*
  Unanswered queries are structured for logging so Solstice's technical writers can identify documentation gaps.

---

## 3. High-Level Architecture (Planned)
*(Detailed component breakdown, retrieval mathematics, and data flow to be completed in the next phase)*

---

## 4. Evaluation & Verification on Benchmark Questions (Planned)
*(Mapping and expected results across Q-1 through Q-5)*

---

## 5. Trade-Offs, Limitations & Production Scaling (Planned)
*(Synonym handling, multi-hop reasoning, role-based access control, and hybrid vector expansion)*
