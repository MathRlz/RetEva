# Dataset Types Guide

This guide explains the different dataset types supported by the evaluator and how to choose the right type for your evaluation task.

## Overview

Dataset types determine:
- Which metrics are computed
- Which pipeline components are required
- How results are interpreted

The evaluator automatically selects appropriate metrics based on the dataset type.

## Dataset Type Classification

### 1. Audio Query Retrieval

**Use case:** Speech-based information retrieval (e.g., medical voice queries → PubMed articles)

**Input:**
- Audio files (queries/questions)
- Text corpus (documents to search)

**Output:**
- Retrieved document IDs
- Relevance scores

**Metrics Computed:**
- **Primary:** MRR, NDCG@10, Recall@10, WER, CER
- **Secondary:** Recall@5, Recall@20, Precision@10, MAP

**Required Components:**
- ASR pipeline (audio → text)
- Text embedding pipeline (text → embeddings)
- Retrieval pipeline (search)

**Example:**
```yaml
data:
  dataset_type: audio_query_retrieval
  questions_path: medical_audio_queries/
  corpus_path: pubmed_corpus.json
```

**When to use:**
- Voice-based search systems
- Spoken question answering
- Medical speech retrieval
- Conversational IR systems

---

### 2. Text Query Retrieval

**Use case:** Text-based information retrieval (e.g., PubMed QA)

**Input:**
- Text queries/questions
- Text corpus (documents to search)

**Output:**
- Retrieved document IDs
- Relevance scores

**Metrics Computed:**
- **Primary:** MRR, NDCG@10, Recall@10
- **Secondary:** MAP, Recall@5, Precision@10, NDCG@5

**Required Components:**
- Text embedding pipeline (text → embeddings)
- Retrieval pipeline (search)

**Example:**
```yaml
data:
  dataset_type: text_query_retrieval
  questions_path: pubmed_qa_questions.json
  corpus_path: pubmed_corpus.json
```

**When to use:**
- Traditional IR benchmarks
- Question answering datasets
- Document retrieval tasks
- No audio involved

**Note:** Can be combined with TTS to synthesize audio for speech retrieval evaluation.

---

### 3. Audio Transcription

**Use case:** Automatic speech recognition (ASR) evaluation

**Input:**
- Audio files
- Reference transcriptions

**Output:**
- Predicted transcriptions

**Metrics Computed:**
- **Primary:** WER (Word Error Rate), CER (Character Error Rate)
- **Secondary:** None

**Required Components:**
- ASR pipeline only

**Example:**
```yaml
data:
  dataset_type: audio_transcription
  audio_dir: medical_audio/
  transcripts_file: transcripts.json

model:
  pipeline_mode: asr_only
```

**When to use:**
- ASR benchmarking
- Transcription accuracy testing
- No retrieval component
- Testing medical/domain-specific ASR

---

### 4. Question Answering

**Use case:** Extractive or abstractive question answering

**Input:**
- Questions (text or audio)
- Corpus with answers

**Output:**
- Retrieved documents or extracted answers

**Metrics Computed:**
- **Primary:** MRR, NDCG@10, Recall@10
- **Secondary:** MAP, Precision@5

**Required Components:**
- Text/Audio embedding (depending on input)
- Retrieval pipeline
- Optional: Answer extraction

**Example:**
```yaml
data:
  dataset_type: question_answering
  questions_path: squad_questions.json
  corpus_path: wikipedia_passages.json
```

**When to use:**
- SQuAD-style datasets
- Extractive QA tasks
- Document-level QA

---

### 5. Multimodal QA

**Use case:** Combined audio and text question answering

**Input:**
- Audio queries
- Text context
- Multimodal corpus

**Output:**
- Retrieved documents
- Transcriptions

**Metrics Computed:**
- **Primary:** MRR, NDCG@10, Recall@10, WER
- **Secondary:** MAP, Recall@5, CER

**Required Components:**
- ASR pipeline
- Text embedding pipeline
- Audio embedding pipeline (optional)
- Retrieval pipeline

**Example:**
```yaml
data:
  dataset_type: multimodal_qa
  questions_path: multimodal_questions.json
  corpus_path: multimodal_corpus.json

model:
  pipeline_mode: audio_text_retrieval
```

**When to use:**
- Cross-modal retrieval
- Multimodal datasets
- Combined text + audio queries

---

### 6. Passage Ranking

**Use case:** Reranking passages by relevance

**Input:**
- Query
- Candidate passages (pre-retrieved)

**Output:**
- Ranked passages

**Metrics Computed:**
- **Primary:** NDCG@10, MRR, MAP
- **Secondary:** NDCG@5, NDCG@20, Recall@10

**Required Components:**
- Text embedding or reranker
- Retrieval pipeline

**Example:**
```yaml
data:
  dataset_type: passage_ranking
  questions_path: msmarco_queries.json
  corpus_path: msmarco_passages.json

vector_db:
  reranker_enabled: true
```

**When to use:**
- MS MARCO
- TREC datasets
- Reranking evaluation

---

## Automatic vs. Manual Dataset Type Selection

### Automatic (Recommended)

Leave `dataset_type` empty or null:

```yaml
data:
  dataset_type:  # Auto-detect
  questions_path: questions.json
  corpus_path: corpus.json
```

The evaluator will:
- Infer type from pipeline mode
- Select all applicable metrics
- Use default metric set

### Manual (Explicit)

Specify dataset type for focused evaluation:

```yaml
data:
  dataset_type: text_query_retrieval  # Explicit
  questions_path: questions.json
  corpus_path: corpus.json
```

Benefits:
- Only relevant metrics computed
- Clearer evaluation intent
- Faster evaluation (fewer metrics)

---

## Metric Selection Reference

| Dataset Type | WER/CER | MRR | NDCG | Recall@K | MAP | Precision@K |
|--------------|---------|-----|------|----------|-----|-------------|
| **Audio Query Retrieval** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Text Query Retrieval** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Audio Transcription** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Question Answering** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Multimodal QA** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Passage Ranking** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## Examples by Use Case

### Example 1: Medical Voice Search

**Goal:** Evaluate spoken medical queries → PubMed articles

```yaml
data:
  dataset_type: audio_query_retrieval
  audio_dir: medical_queries/
  transcripts_file: transcripts.json
  corpus_path: pubmed_corpus.json

model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_adapter_path: medical-whisper-adapter/  # Domain adaptation
  text_emb_model_type: jina_v4
```

**Metrics:** MRR, NDCG@10, WER (with medical terminology)

---

### Example 2: Text-Only PubMed QA

**Goal:** Evaluate text queries on PubMed corpus

```yaml
data:
  dataset_type: text_query_retrieval
  questions_path: pubmed_qa_questions.json
  corpus_path: pubmed_abstracts.json

model:
  pipeline_mode: text_retrieval  # Text only, no ASR
  text_emb_model_type: jina_v4
```

**Metrics:** MRR, NDCG@10, Recall@10 (no WER/CER)

---

### Example 3: ASR Benchmarking

**Goal:** Test medical ASR accuracy

```yaml
data:
  dataset_type: audio_transcription
  audio_dir: medical_speech/
  transcripts_file: ground_truth.json

model:
  pipeline_mode: asr_only  # No retrieval
  asr_model_type: whisper
  asr_adapter_path: medical-adapter/
```

**Metrics:** WER, CER only

---

### Example 4: Text + TTS = Audio Retrieval

**Goal:** Synthesize audio from text questions for speech retrieval

```yaml
data:
  dataset_type: audio_query_retrieval  # Will compute WER
  questions_path: text_questions.json
  corpus_path: corpus.json

model:
  pipeline_mode: asr_text_retrieval

audio_synthesis:
  enabled: true  # Convert text to audio
  provider: piper
  voice: en_US-lessac-medium
```

**Workflow:**
1. Text questions synthesized to audio
2. Audio processed by ASR
3. ASR output embedded
4. Retrieval performed
5. Metrics: MRR, NDCG, WER (on synthesized vs ASR)

---

## Choosing the Right Dataset Type

**Decision tree:**

```
Do you have audio input?
├─ Yes
│  ├─ Do you need retrieval?
│  │  ├─ Yes → Audio Query Retrieval
│  │  └─ No → Audio Transcription
│  └─ Text input?
│     └─ Yes → Multimodal QA
└─ No (text only)
   ├─ Passage ranking task?
   │  ├─ Yes → Passage Ranking
   │  └─ No → Text Query Retrieval
   └─ QA with answer extraction?
      └─ Yes → Question Answering
```

---

## Advanced: Custom Metric Selection

Override automatic metrics in config:

```yaml
data:
  dataset_type: audio_query_retrieval

# Explicitly specify metrics (advanced)
metrics:
  primary:
    - mrr
    - ndcg@10
    - wer
  secondary:
    - recall@5
    - map
```

---

## See Also

- [TTS Setup Guide](tts-setup.md) - Synthesize audio from text
- [Configuration Reference](../README.md#configuration)
- [Metrics Documentation](metrics.md)
