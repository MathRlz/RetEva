# Query Optimization

## Overview

Query optimization techniques transform user queries before retrieval to improve search quality. The evaluator framework implements several state-of-the-art query optimization methods that leverage Large Language Models (LLMs) to reformulate, expand, or decompose queries for better retrieval performance.

## Why Query Optimization?

User queries are often:
- **Too short or ambiguous**: "What causes chest pain?"
- **Poorly phrased**: Missing key medical terminology
- **Too complex**: Multiple questions combined
- **Vocabulary mismatch**: Query terms differ from document terms

Query optimization addresses these issues by:
- Reformulating queries with domain-specific terminology
- Expanding queries with synonyms and related concepts
- Decomposing complex queries into sub-queries
- Generating hypothetical answers to improve embedding similarity

## Available Methods

### 1. Query Rewriting (Iterative Refinement)

Rewrites queries using retrieved context to progressively improve results.

**How it works**:
1. Perform initial retrieval with original query
2. Pass query + retrieved docs to LLM
3. LLM generates improved query
4. Retrieve with improved query
5. Optionally repeat for N iterations

**When to use**:
- When initial queries are vague or poorly phrased
- For medical queries that need terminology normalization
- When you want iterative improvement

**Configuration**:
```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 2
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.7
  rewrite_prompt: |
    Given the query: "{query}"
    And the following retrieved documents:
    {context}
    
    Rewrite the query to be more specific and use appropriate medical terminology.
    Return only the rewritten query, nothing else.
```

**Example**:
- Original: "heart problem symptoms"
- Rewritten: "What are the clinical manifestations of myocardial infarction and other cardiovascular diseases?"

### 2. HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer to the query, then uses the answer's embedding for retrieval.

**How it works**:
1. LLM generates a hypothetical answer to the query
2. Embed the hypothetical answer
3. Use answer embedding to search for similar documents
4. Documents similar to the hypothetical answer are likely to answer the query

**When to use**:
- When documents contain answers, not questions
- For bridging query-document vocabulary gap
- When you have a powerful LLM available

**Configuration**:
```yaml
query_optimization:
  enabled: true
  method: "hyde"
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.8
  hyde_prompt: |
    Write a detailed answer to the following medical question:
    "{query}"
    
    Provide a comprehensive answer as if you were a medical professional.
```

**Example**:
- Query: "What is the treatment for type 2 diabetes?"
- Hypothetical Answer: "Type 2 diabetes is typically treated with lifestyle modifications including diet and exercise, along with medications such as metformin, sulfonylureas, or insulin if needed..."
- Retrieval uses the answer embedding, finding documents with similar treatment information

### 3. Query Decomposition

Breaks complex queries into simpler sub-queries, retrieves for each, then combines results.

**How it works**:
1. LLM analyzes query complexity
2. Decompose into independent sub-queries
3. Retrieve documents for each sub-query
4. Combine and deduplicate results

**When to use**:
- For multi-part questions
- When queries contain multiple topics
- For comprehensive information gathering

**Configuration**:
```yaml
query_optimization:
  enabled: true
  method: "decompose"
  max_sub_queries: 3
  llm:
    provider: "openai"
    model: "gpt-4"
  decompose_prompt: |
    Decompose the following complex medical query into simpler sub-queries:
    "{query}"
    
    Return a JSON list of sub-queries: ["sub-query 1", "sub-query 2", ...]
```

**Example**:
- Original: "What are the causes, symptoms, and treatments for COPD?"
- Sub-queries:
  1. "What causes COPD?"
  2. "What are the symptoms of COPD?"
  3. "What are the treatment options for COPD?"

### 4. Multi-Query Generation

Generates multiple paraphrased versions of the query to improve recall.

**How it works**:
1. LLM generates N alternative phrasings
2. Retrieve documents for each variant
3. Combine results using rank fusion

**When to use**:
- To improve recall (find more relevant documents)
- When there are multiple valid ways to express the query
- For robustness to phrasing variations

**Configuration**:
```yaml
query_optimization:
  enabled: true
  method: "multi_query"
  num_queries: 3
  fusion_method: "rrf"  # Reciprocal Rank Fusion
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
```

**Example**:
- Original: "diabetes complications"
- Variants:
  1. "What are the long-term complications of diabetes mellitus?"
  2. "Diabetic complications and comorbidities"
  3. "Health problems caused by chronic diabetes"

## Configuration Guide

### Basic Setup with OpenAI

```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 2
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}"  # From environment variable
    temperature: 0.7
    max_tokens: 256
```

### Using Local LLMs (vLLM)

```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  llm:
    provider: "vllm"
    model: "meta-llama/Llama-2-7b-chat-hf"
    api_base: "http://localhost:8000/v1"
    temperature: 0.7
```

### Using Ollama

```yaml
query_optimization:
  enabled: true
  method: "hyde"
  llm:
    provider: "ollama"
    model: "llama2:7b"
    api_base: "http://localhost:11434"
```

### Custom Prompts

Customize prompts for your domain:

```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  rewrite_prompt: |
    You are a medical information specialist. Rewrite the following query
    to use precise medical terminology from the SNOMED CT vocabulary:
    
    Query: "{query}"
    
    Context from retrieved documents:
    {context}
    
    Rewritten query (medical terminology only):
```

## Cost Management

### 1. Enable Caching

```yaml
query_optimization:
  enabled: true
  cache_llm_calls: true
  cache_ttl: 86400  # 24 hours
```

This caches LLM responses to avoid redundant API calls for the same queries.

### 2. Use Cheaper Models

For query rewriting, GPT-3.5-turbo is often sufficient:

```yaml
query_optimization:
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"  # Cheaper than gpt-4
```

### 3. Limit Iterations

```yaml
query_optimization:
  method: "rewrite"
  num_iterations: 1  # Start with 1, increase if needed
```

### 4. Sample During Development

Only optimize a subset of queries during experimentation:

```yaml
query_optimization:
  enabled: true
  sample_rate: 0.1  # Only optimize 10% of queries
```

### 5. Use Local LLMs

Deploy local models to eliminate API costs:

```yaml
query_optimization:
  llm:
    provider: "ollama"
    model: "llama2:7b"  # Free, runs locally
```

## Medical Query Best Practices

### 1. Medical Terminology Normalization

```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  rewrite_prompt: |
    Rewrite this query using standard medical terminology (ICD-10, SNOMED CT):
    "{query}"
    
    Use formal medical terms, not colloquial language.
    Rewritten query:
```

### 2. Abbreviation Expansion

```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  rewrite_prompt: |
    Expand abbreviations and rewrite this medical query:
    "{query}"
    
    Replace abbreviations with full terms (e.g., MI -> myocardial infarction).
    Expanded query:
```

### 3. Multi-Aspect Medical Queries

For comprehensive medical information:

```yaml
query_optimization:
  enabled: true
  method: "decompose"
  decompose_prompt: |
    Decompose this medical query into separate aspects:
    - Etiology (causes)
    - Pathophysiology
    - Clinical presentation (symptoms/signs)
    - Diagnosis
    - Treatment
    - Prognosis
    
    Query: "{query}"
    
    Return relevant sub-queries as JSON list.
```

### 4. Evidence-Based Query Enhancement

```yaml
query_optimization:
  enabled: true
  method: "rewrite"
  rewrite_prompt: |
    Enhance this query for evidence-based medical literature search:
    "{query}"
    
    Add relevant PICO elements (Population, Intervention, Comparison, Outcome).
    Enhanced query:
```

## Combining Multiple Techniques

### Sequential Pipeline

```python
# Pseudo-code example
query = "heart problems"
query = rewrite(query)  # -> "myocardial infarction symptoms"
sub_queries = decompose(query)  # -> ["MI causes", "MI symptoms", "MI diagnosis"]
for sub_q in sub_queries:
    hyde_answer = generate_hyde(sub_q)
    results.extend(retrieve(hyde_answer))
```

Configuration for sequential optimization:

```yaml
query_optimization:
  enabled: true
  pipeline:
    - method: "rewrite"
      num_iterations: 1
    - method: "decompose"
      max_sub_queries: 3
    - method: "hyde"
      per_sub_query: true
```

## Performance Considerations

### Latency

- **Query Rewriting**: +0.5-2s per LLM call
- **HyDE**: +1-3s per LLM call (generates longer text)
- **Decomposition**: +0.5-2s per LLM call
- **Multi-Query**: +0.5-2s + (N-1) retrieval calls

**Mitigation**:
- Use faster models (gpt-3.5-turbo vs gpt-4)
- Enable caching
- Use local LLMs with GPU acceleration
- Parallel sub-query retrieval

### API Costs

Approximate costs with OpenAI (GPT-3.5-turbo, $0.50/$1.50 per 1M tokens):

- **Query Rewriting**: ~$0.001 per query (input: ~200 tokens, output: ~50 tokens)
- **HyDE**: ~$0.003 per query (output: ~500 tokens)
- **Decomposition**: ~$0.001 per query
- **Multi-Query**: ~$0.002 per query (3 variants)

For 1000 queries with rewriting: **~$1.00**

### Quality vs Cost Trade-off

```yaml
# High quality (expensive)
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 3
  llm:
    model: "gpt-4"

# Balanced (recommended)
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 2
  llm:
    model: "gpt-3.5-turbo"

# Low cost (local)
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 1
  llm:
    provider: "ollama"
    model: "llama2:7b"
```

## Troubleshooting

### Issue: LLM calls timing out

**Solutions**:
1. Increase timeout: `llm.timeout: 30`
2. Use faster model: `model: "gpt-3.5-turbo"`
3. Reduce max_tokens: `max_tokens: 128`

### Issue: Poor rewriting quality

**Solutions**:
1. Improve prompt with examples (few-shot learning)
2. Use more powerful model: `model: "gpt-4"`
3. Increase temperature for creativity: `temperature: 0.9`
4. Provide more context in prompt

### Issue: High API costs

**Solutions**:
1. Enable caching: `cache_llm_calls: true`
2. Use cheaper model: `model: "gpt-3.5-turbo"`
3. Reduce iterations: `num_iterations: 1`
4. Sample queries during dev: `sample_rate: 0.1`
5. Switch to local LLM (Ollama)

### Issue: Query decomposition too aggressive

**Solutions**:
1. Reduce max sub-queries: `max_sub_queries: 2`
2. Adjust prompt to be more conservative
3. Add filtering to remove redundant sub-queries

## Examples

### Example 1: Medical Query Rewriting

```yaml
# configs/medical_rewrite.yaml
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 2
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.7
  rewrite_prompt: |
    Rewrite this medical query using precise clinical terminology:
    "{query}"
    
    Retrieved context:
    {context}
    
    Requirements:
    - Use standard medical terms
    - Expand abbreviations
    - Add relevant qualifiers
    - Keep query concise
    
    Rewritten query:
```

### Example 2: HyDE for Medical QA

```yaml
# configs/hyde_medical.yaml
query_optimization:
  enabled: true
  method: "hyde"
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.8
  hyde_prompt: |
    You are a medical expert. Provide a detailed answer to:
    "{query}"
    
    Include:
    - Clinical definition
    - Key medical facts
    - Standard treatment approaches
    - Relevant medical terminology
    
    Answer:
```

### Example 3: Complex Query Decomposition

```yaml
# configs/decompose_complex.yaml
query_optimization:
  enabled: true
  method: "decompose"
  max_sub_queries: 4
  llm:
    provider: "openai"
    model: "gpt-4"
  decompose_prompt: |
    Break down this complex medical query into simpler sub-queries:
    "{query}"
    
    Each sub-query should address one specific aspect.
    Return as JSON: ["query1", "query2", ...]
```

### Example 4: Local LLM (Cost-Free)

```yaml
# configs/local_llm_optimization.yaml
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 1
  llm:
    provider: "ollama"
    model: "meditron:7b"  # Medical-domain LLM
    api_base: "http://localhost:11434"
    temperature: 0.7
```

## Integration with Other Features

Query optimization works with:

### Hybrid Retrieval

```yaml
query_optimization:
  enabled: true
  method: "rewrite"

vector_db:
  retrieval_mode: "hybrid"
  hybrid_alpha: 0.7  # Optimized query used for both dense + BM25
```

### Embedding Fusion

```yaml
query_optimization:
  enabled: true
  method: "hyde"

vector_db:
  embedding_fusion:
    enabled: true
    method: "weighted"
    # HyDE answer embedded with fusion
```

### Reranking

```yaml
query_optimization:
  enabled: true
  method: "multi_query"

vector_db:
  use_reranking: true
  # Multiple query variants -> diverse results -> reranking
```

## See Also

- [Embedding Fusion](embedding_fusion.md) - Combine audio and text embeddings
- [Advanced RAG](advanced_rag.md) - Multi-vector and expansion techniques
- [LLM Judge](llm_judge.md) - Evaluate with LLM-based metrics
- [Configuration Reference](configuration_reference.md) - Complete config options
- [Tutorial Notebook 16](../../notebooks/16_query_optimization_techniques.ipynb) - Interactive demos
