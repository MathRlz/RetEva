# LLM-as-Judge Evaluation

## Overview

LLM-as-Judge uses Large Language Models to evaluate retrieval quality through natural language assessment. Instead of relying solely on position-based metrics (MRR, NDCG), LLMs can assess multiple dimensions of relevance, provide qualitative feedback, and catch nuances that traditional metrics miss.

The evaluator framework implements sophisticated LLM judging with multi-aspect evaluation, calibration, consistency testing, and support for both API-based and local models.

## Why LLM-as-Judge?

### Limitations of Traditional Metrics

- **Position-dependent**: MRR, NDCG only consider rank position
- **Binary relevance**: Document is either relevant or not (no gradations)
- **No quality assessment**: Can't distinguish between "somewhat relevant" and "highly relevant"
- **Missing aspects**: Don't evaluate factuality, completeness, clarity

### Advantages of LLM Judging

- **Multi-dimensional**: Evaluate multiple aspects simultaneously
- **Nuanced scoring**: Capture gradations of relevance (1-5 scale)
- **Qualitative feedback**: Provide explanations for scores
- **Domain-aware**: Can apply medical domain knowledge
- **Flexible criteria**: Easily adapt evaluation criteria

## Multi-Aspect Judging

### Available Aspects

The framework evaluates multiple dimensions of quality:

```yaml
judge:
  enabled: true
  aspects:
    - name: "relevance"
      weight: 0.3
      description: "How relevant is this document to the query?"
    
    - name: "accuracy"
      weight: 0.25
      description: "Is the information medically accurate?"
    
    - name: "completeness"
      weight: 0.2
      description: "Does the document fully address the query?"
    
    - name: "clarity"
      weight: 0.15
      description: "Is the information clearly presented?"
    
    - name: "factuality"
      weight: 0.1
      description: "Are medical facts correct and verifiable?"
```

### Custom Aspects

Define domain-specific aspects:

```yaml
judge:
  aspects:
    - name: "clinical_applicability"
      weight: 0.3
      description: "Can this information be applied in clinical practice?"
      scoring_guidelines: |
        5: Directly actionable clinical guidance
        4: Useful clinical information
        3: Some clinical relevance
        2: Limited clinical value
        1: No clinical applicability
    
    - name: "evidence_quality"
      weight: 0.3
      description: "Quality of evidence presented"
      scoring_guidelines: |
        5: Systematic review/meta-analysis
        4: Randomized controlled trial
        3: Cohort study
        2: Case series
        1: Expert opinion only
```

## Score Aggregation

### Weighted Average

Default method, combines aspect scores with weights:

```yaml
judge:
  aggregation_method: "weighted_average"
  aspects:
    - name: "relevance"
      weight: 0.5  # Most important
    - name: "accuracy"
      weight: 0.3
    - name: "clarity"
      weight: 0.2
```

### Minimum Score

Require minimum threshold for all aspects:

```yaml
judge:
  aggregation_method: "minimum"
  # Final score = min(all aspect scores)
  # Ensures no critical aspect is too low
```

### Harmonic Mean

Penalize low scores more heavily:

```yaml
judge:
  aggregation_method: "harmonic_mean"
  # Sensitive to low scores
  # Good for quality control
```

### Custom Aggregation

Define custom scoring logic:

```yaml
judge:
  aggregation_method: "custom"
  aggregation_function: |
    def aggregate(scores):
        # Must have high relevance and accuracy
        if scores['relevance'] < 3 or scores['accuracy'] < 3:
            return 0
        # Otherwise weighted average
        return 0.5 * scores['relevance'] + 0.3 * scores['accuracy'] + 0.2 * scores['completeness']
```

## Configuration Guide

### Basic Setup with OpenAI

```yaml
judge:
  enabled: true
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.3  # Lower for consistent scoring
  
  aspects:
    - name: "relevance"
      weight: 0.6
    - name: "accuracy"
      weight: 0.4
  
  aggregation_method: "weighted_average"
```

### Using GPT-3.5-turbo (Cost-Effective)

```yaml
judge:
  enabled: true
  provider: "openai"
  model: "gpt-3.5-turbo"
  # Much cheaper, often sufficient
  aspects:
    - name: "relevance"
      weight: 1.0
  # Use fewer aspects to save tokens
```

### Local LLM with vLLM

```yaml
judge:
  enabled: true
  provider: "vllm"
  model: "meta-llama/Llama-2-13b-chat-hf"
  api_base: "http://localhost:8000/v1"
  temperature: 0.3
  max_tokens: 256
```

### Ollama for Free Local Judging

```yaml
judge:
  enabled: true
  provider: "ollama"
  model: "llama2:13b"
  api_base: "http://localhost:11434"
  temperature: 0.3
```

### Medical Domain Judging

```yaml
judge:
  enabled: true
  provider: "openai"
  model: "gpt-4"
  
  system_prompt: |
    You are a medical information specialist evaluating the quality of 
    medical literature retrieval results. Apply your knowledge of clinical 
    medicine and evidence-based practice to assess relevance and accuracy.
  
  aspects:
    - name: "medical_relevance"
      weight: 0.4
      description: "Relevance to the medical query"
    
    - name: "clinical_accuracy"
      weight: 0.3
      description: "Medical accuracy and current best practices"
    
    - name: "evidence_level"
      weight: 0.2
      description: "Quality of evidence (RCT > observational > case report)"
    
    - name: "safety_considerations"
      weight: 0.1
      description: "Are safety concerns properly addressed?"
```

## Few-Shot Learning

### Providing Examples

Improve judging quality with few-shot examples:

```yaml
judge:
  enabled: true
  few_shot_examples:
    - query: "What are the symptoms of type 2 diabetes?"
      document: "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, fatigue, and blurred vision."
      scores:
        relevance: 5
        accuracy: 5
        completeness: 4
      explanation: "Highly relevant and accurate list of common T2D symptoms. Completeness is 4 because it doesn't mention less common symptoms."
    
    - query: "What causes heart attacks?"
      document: "Exercise is good for your health."
      scores:
        relevance: 1
        accuracy: 4
        completeness: 1
      explanation: "While accurate, this is not relevant to the query about heart attack causes."
    
    - query: "Treatment for hypertension"
      document: "Hypertension treatment includes lifestyle modifications and medications like ACE inhibitors, ARBs, and diuretics."
      scores:
        relevance: 5
        accuracy: 5
        completeness: 4
      explanation: "Highly relevant and accurate. Slightly incomplete as it doesn't detail dosing or monitoring."
```

### Medical Domain Examples

```yaml
judge:
  few_shot_examples:
    - query: "Evidence for metformin in type 2 diabetes"
      document: "Multiple RCTs show metformin reduces HbA1c by 1-2% with low hypoglycemia risk."
      scores:
        medical_relevance: 5
        clinical_accuracy: 5
        evidence_level: 5
      explanation: "Cites RCT evidence (highest level) with specific, accurate clinical data."
    
    - query: "Aspirin for primary prevention"
      document: "Aspirin is good for your heart."
      scores:
        medical_relevance: 2
        clinical_accuracy: 2
        evidence_level: 1
      explanation: "Vague and overgeneralized. Primary prevention guidelines are nuanced; this lacks specificity."
```

## Calibration and Consistency

### Calibration Testing

Test judge against known cases:

```yaml
judge:
  calibration:
    enabled: true
    test_cases:
      - query: "COVID-19 symptoms"
        document: "COVID-19 symptoms include fever, cough, and shortness of breath."
        expected_score: 5
        tolerance: 0.5
      
      - query: "COVID-19 symptoms"
        document: "The weather is sunny today."
        expected_score: 1
        tolerance: 0.5
```

Run calibration:
```bash
python -m evaluator.evaluation.calibrate_judge --config config.yaml
```

### Consistency Testing

Test same query-document pair multiple times:

```yaml
judge:
  consistency_test:
    enabled: true
    num_trials: 5
    max_std_dev: 0.5  # Maximum acceptable standard deviation
```

Example:
```python
# Test consistency
query = "diabetes treatment"
document = "..."

scores = []
for i in range(5):
    score = judge.evaluate(query, document)
    scores.append(score)

mean_score = np.mean(scores)
std_dev = np.std(scores)

if std_dev > 0.5:
    print(f"Warning: High variance in scores (std={std_dev:.2f})")
```

### Temperature Tuning

Lower temperature for more consistent scoring:

```yaml
judge:
  temperature: 0.1  # More consistent, less varied
  # vs
  temperature: 0.7  # More varied, potentially more creative
```

## Cost Management

### API Cost Estimation

Approximate costs for OpenAI:

**GPT-4**:
- Input: ~400 tokens (query + document + prompt)
- Output: ~100 tokens (scores + explanation)
- Cost per judgment: ~$0.02
- 1000 judgments: ~$20

**GPT-3.5-turbo**:
- Same token counts
- Cost per judgment: ~$0.001
- 1000 judgments: ~$1

**GPT-4-turbo**:
- Cost per judgment: ~$0.005
- 1000 judgments: ~$5

### Cost Reduction Strategies

#### 1. Sample Evaluation

Only judge a subset of results:

```yaml
judge:
  enabled: true
  sample_rate: 0.1  # Only judge 10% of documents
  sampling_strategy: "stratified"  # Ensure diverse sample
```

#### 2. Use Cheaper Models

```yaml
judge:
  provider: "openai"
  model: "gpt-3.5-turbo"  # 20x cheaper than GPT-4
  # Often sufficient for relevance judging
```

#### 3. Reduce Aspects

```yaml
judge:
  aspects:
    - name: "relevance"
      weight: 1.0
  # Single aspect = fewer tokens
```

#### 4. Disable Explanations

```yaml
judge:
  require_explanation: false
  # Saves ~50 output tokens per judgment
```

#### 5. Enable Caching

```yaml
judge:
  cache_results: true
  cache_ttl: 86400  # 24 hours
  # Avoid re-judging same query-doc pairs
```

#### 6. Use Local LLMs

```yaml
judge:
  provider: "ollama"
  model: "llama2:13b"
  # Free, no API costs
```

#### 7. Batch Processing

```yaml
judge:
  batch_size: 10
  # Send multiple judgments in one API call
  # Reduces per-request overhead
```

## Local LLM Support

### vLLM Setup

1. Install vLLM:
```bash
pip install vllm
```

2. Start server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-chat-hf \
    --port 8000
```

3. Configure judge:
```yaml
judge:
  provider: "vllm"
  model: "meta-llama/Llama-2-13b-chat-hf"
  api_base: "http://localhost:8000/v1"
```

### Ollama Setup

1. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
```

2. Pull model:
```bash
ollama pull llama2:13b
```

3. Configure judge:
```yaml
judge:
  provider: "ollama"
  model: "llama2:13b"
  api_base: "http://localhost:11434"
```

### Medical Domain LLMs

For medical judging, consider specialized models:

```yaml
judge:
  provider: "vllm"
  model: "epfl-llm/meditron-7b"  # Medical domain LLM
  api_base: "http://localhost:8000/v1"
```

Or:
```yaml
judge:
  provider: "ollama"
  model: "meditron:7b"
```

## Best Practices

### 1. Start with GPT-4, Calibrate, Then Downgrade

```yaml
# Development: Use GPT-4 for quality
judge:
  model: "gpt-4"

# After calibration: Test GPT-3.5
judge:
  model: "gpt-3.5-turbo"

# Compare: If quality drop is acceptable, use GPT-3.5
```

### 2. Use Few-Shot Examples

Always provide 2-5 examples:
```yaml
judge:
  few_shot_examples:
    - ... # High relevance example
    - ... # Low relevance example
    - ... # Medium relevance example
```

### 3. Focus on Critical Aspects

Don't over-engineer:
```yaml
# Good: 2-3 key aspects
judge:
  aspects:
    - relevance (0.6)
    - accuracy (0.4)

# Avoid: Too many aspects
judge:
  aspects:  # 7 aspects may dilute focus
    - relevance, accuracy, completeness, clarity, 
      factuality, currency, specificity
```

### 4. Test Consistency

Always run consistency tests:
```bash
python -m evaluator.evaluation.test_judge_consistency \
    --config config.yaml \
    --num_trials 10
```

### 5. Monitor Costs

Track API usage:
```python
from evaluator.evaluation import JudgeCostTracker

tracker = JudgeCostTracker()
results = evaluate(config, cost_tracker=tracker)

print(f"Total judgments: {tracker.num_judgments}")
print(f"Total tokens: {tracker.total_tokens}")
print(f"Estimated cost: ${tracker.estimated_cost:.2f}")
```

### 6. Medical Domain Prompts

Use domain-specific prompts:
```yaml
judge:
  system_prompt: |
    You are evaluating medical literature retrieval. Consider:
    - Clinical relevance and applicability
    - Evidence quality (prefer RCTs, systematic reviews)
    - Safety and contraindications
    - Current medical guidelines
```

## Integration Examples

### With Hybrid Retrieval

```yaml
vector_db:
  retrieval_mode: "hybrid"
  hybrid_alpha: 0.7

judge:
  enabled: true
  aspects:
    - name: "relevance"
      weight: 1.0
  # Evaluate quality of hybrid results
```

### With Query Optimization

```yaml
query_optimization:
  enabled: true
  method: "rewrite"

judge:
  enabled: true
  # Evaluate if optimization improved relevance
  track_by_optimization_method: true
```

### With Reranking

```yaml
vector_db:
  use_reranking: true
  reranking_model: "cross-encoder"

judge:
  enabled: true
  # Compare judge scores before/after reranking
  evaluate_reranking_quality: true
```

## Troubleshooting

### Issue: Inconsistent scores

**Solutions**:
1. Lower temperature: `temperature: 0.1`
2. Add more few-shot examples
3. Make scoring guidelines more specific
4. Use GPT-4 instead of GPT-3.5

### Issue: High API costs

**Solutions**:
1. Sample evaluation: `sample_rate: 0.1`
2. Use GPT-3.5-turbo
3. Reduce aspects to 1-2
4. Disable explanations
5. Switch to local LLM

### Issue: Judge too lenient/strict

**Solutions**:
1. Adjust few-shot examples to show desired strictness
2. Modify scoring guidelines
3. Test with calibration set
4. Try different model

### Issue: Slow evaluation

**Solutions**:
1. Enable batching: `batch_size: 10`
2. Parallel requests: `max_concurrent: 5`
3. Use faster model: GPT-3.5 or local LLM
4. Reduce max_tokens: `max_tokens: 128`

## When to Use LLM Judge

### ✅ Use LLM Judge When:

- Evaluating nuanced relevance
- Need qualitative feedback
- Traditional metrics insufficient
- Multi-dimensional quality assessment
- Medical/specialized domain evaluation

### ❌ Don't Use LLM Judge When:

- Budget is very limited
- Need exact reproducibility
- Simple binary relevance
- Large-scale evaluation (>100k queries)
- Real-time evaluation required

## See Also

- [Query Optimization](query_optimization.md) - Optimize queries with LLMs
- [Configuration Reference](configuration_reference.md) - Complete config options
- [Tutorial Notebook 18](../../notebooks/18_llm_judge_evaluation.ipynb) - Interactive judging examples
- [Tutorial Notebook 19](../../notebooks/19_full_rag_ablation_study.ipynb) - Compare metrics vs LLM judge
