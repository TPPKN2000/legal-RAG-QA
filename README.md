# **LegalRAG**  
Legal case outcome prediction requires more than simple text classification. A strong system must understand the legal dispute, identify the claims of the parties, retrieve relevant factual evidence from the case record, retrieve applicable legal provisions, and reason about how the court is likely to resolve the dispute.


This task aims to encourage research on:

- Vietnamese legal judgment understanding.

- Retrieval-augmented legal reasoning.

- Agentic interaction with legal APIs.

- Evidence-grounded legal prediction.

- Vietnamese legal corpus retrieval.

- Transparent and verifiable legal AI systems.


The task setting reflects a realistic legal AI scenario: a system receives an initial case description, then must actively retrieve additional information before making a prediction.

So, this is a Retrieval-Augmented Generation (RAG) system for it, built with **FastAPI**, **Pinecone**, and local **Qwen/Qwen3.5-0.8B**.  

## **Overview**  

For each test instance, participants are given a short natural-language query describing the dispute. The query includes the main parties, the disputed legal relationship or asset, a brief summary of the plaintiff's claim and the defendant's position, and a question asking whether the plaintiff or the defendant is more likely to win.


Participants must build a system that:


1. Reads the provided case query.

2. Calls the official Case Content API to retrieve relevant case segments.

3. Retrieves relevant legal provisions from the provided law corpus.

4. Predicts the final outcome of the case.

5. Submits the predicted outcome together with supporting case evidence and legal provisions.


The competition contains only one task:

Legal Case Outcome Prediction with Evidence Retrieval 
The expected prediction is one of four labels: 

- A_WIN: The court fully accepts all of the plaintiff's claims.

- PARTIAL_A_WIN: The court partially accepts the plaintiff's claims, and the accepted portion is greater than 50%.

- PARTIAL_B_WIN: The court partially accepts the plaintiff's claims, but the accepted portion is 50% or less.

- B_WIN: The court fully rejects all of the plaintiff's claims.

## **System Architecture & Pipeline**

```txt
                                +-------------------+
                                |    Case Query     |
                                +---------+---------+
                                          |
                                          v
                   +----------------------+----------------------+
                   |      NER Query Rewriter (Electra-Base)      |
                   +----------------------+----------------------+
                                          |
                                          v (Rewritten Query)
                                          |     +-------------------------+
                                          |     |    Case Content API     |
                                          |     +------------+------------+
                                          |                  |
                                          v                  v (Case Evidence)
                   +----------------------+------------------+---+
                   |         HyDE Query Generator (Qwen-2B)      |
                   +----------------------+----------------------+
                                          |
                                          v (HyDE Synthetic Answer)
                   +----------------------+----------------------+
                   |               Fusion Retriever              |
                   |  - BM25 Search (30% weight)                 |
                   |  - Pinecone VNBertLaw Embeddings (70% weight)|
                   +----------------------+----------------------+
                                          |
                                          v (Top 20 Law Candidates)
                   +----------------------+----------------------+
                   |      Vietnamese Reranker (Cross-Encoder)     |
                   +----------------------+----------------------+
                                          |
                                          v (Top 5 Law Provisions)
                   +----------------------+----------------------+
                   |     Legal Case Predictor (Qwen-0.8B)         |
                   |  - Inputs: Query + Evidence + Top 5 Laws     |
                   +----------------------+----------------------+
                                          |
                                          v
                               +----------+----------+
                               |    Final Judgment    |
                               |  - Verdict Label    |
                               |  - Legal Reasoning  |
                               +---------------------+
```

## **Installation**  

### **2. Create a virtual environment**  
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### **3. Install dependencies**  
```bash
pip install -r requirements.txt
```

---  

## **Configuration**  

Create a `.env` file:
```bash
cp .env.example .env
```
and add the following variables:  
```
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_pinecone_index_name
ALQAC_TOKEN=alqac_xxxxxxxxxxxxxxxxxxxxxxxx
```

---  

## TEST PIPELINE
```bash
venv/Scripts/activate
# Load your document on Pinecone
python document_uploader.py
# Test 1 case
python -m test.test_first_case
# Test all
python -m test.test_all
```


## **License**  
This project is open-source under the **MIT License**.  
