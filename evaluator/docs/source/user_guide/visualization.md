# Visualization

Analyze and visualize evaluation results using Jupyter notebooks and the Web UI.

## Jupyter Notebooks

The `notebooks/` directory contains interactive analysis notebooks.

### Installation

```bash
pip install -e .
```

This installs:
- Jupyter, Notebook, IPyWidgets
- Matplotlib, Seaborn, Plotly
- Pandas, SciPy, Kaleido

### Available Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_basic_result_exploration.ipynb` | Load and explore single evaluation results |
| `02_multi_experiment_comparison.ipynb` | Compare multiple experiments with statistics |
| `03_visualization_plots.ipynb` | Create matplotlib/seaborn/plotly visualizations |
| `04_statistical_analysis.ipynb` | Bootstrap CI, significance tests, effect sizes |
| `05_ablation_study_analysis.ipynb` | Ablation analysis and component impact |
| `06_export_publication_tables.ipynb` | Export LaTeX/Excel tables for papers |

### Quick Start

```bash
# Copy results to notebooks directory
cp evaluation_results/*.json notebooks/data/

# Launch Jupyter
cd notebooks
jupyter notebook
```

### Basic Result Exploration

```python
# In notebook: 01_basic_result_exploration.ipynb
from evaluator.visualization import ResultsLoader, plot_metrics

# Load results
loader = ResultsLoader()
results = loader.load("data/whisper_labse.json")

# Display summary
results.summary()

# Plot metrics
plot_metrics(results, metrics=["MRR", "MAP", "NDCG@5", "Recall@5"])
```

### Multi-Experiment Comparison

```python
# In notebook: 02_multi_experiment_comparison.ipynb
from evaluator.visualization import compare_experiments

# Load multiple experiments
experiments = {
    "Whisper-Base + LaBSE": "data/whisper_base_labse.json",
    "Whisper-Large + Jina": "data/whisper_large_jina.json",
    "Audio Direct": "data/audio_only.json"
}

results = loader.load_multiple(experiments)

# Compare
comparison = compare_experiments(results)
comparison.plot_comparison()
comparison.export_table("comparison.csv")
```

### Statistical Analysis

```python
# In notebook: 04_statistical_analysis.ipynb
from evaluator.visualization import StatisticalAnalysis

analysis = StatisticalAnalysis(results)

# Bootstrap confidence intervals
ci = analysis.bootstrap_ci(metric="MRR", n_bootstrap=10000)
print(f"MRR: {ci.mean:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")

# Significance test
p_value = analysis.paired_t_test(
    results1=whisper_base,
    results2=whisper_large,
    metric="MRR"
)
print(f"p-value: {p_value:.4f}")

# Effect size
cohen_d = analysis.cohens_d(results1, results2, metric="MRR")
print(f"Cohen's d: {cohen_d:.4f}")
```

### Visualization Plots

```python
# In notebook: 03_visualization_plots.ipynb
from evaluator.visualization import (
    plot_metric_distribution,
    plot_rank_distribution,
    plot_error_analysis,
    plot_retrieval_heatmap
)

# Metric distribution
plot_metric_distribution(results, metric="MRR")

# Rank distribution
plot_rank_distribution(results)

# Error analysis
plot_error_analysis(
    results,
    group_by="specialty",  # Group errors by metadata field
    top_n=10
)

# Retrieval heatmap
plot_retrieval_heatmap(results, top_k=10)
```

## Web UI

Interactive web interface for visualizing results.

### Installation

```bash
pip install -e .
```

### Launch Web UI

```bash
# From repository root
cd webui
python app.py

# Or use the launcher
python -m evaluator.webui
```

Navigate to `http://localhost:5000`

### Features

1. **Results Dashboard**
   - Upload and view evaluation results
   - Interactive metric plots
   - Filterable result tables

2. **Experiment Comparison**
   - Side-by-side comparison
   - Statistical tests
   - Export comparisons

3. **Sample Exploration**
   - Browse individual samples
   - View predictions vs ground truth
   - Play audio (if available)
   - See transcriptions

4. **Metric Analysis**
   - Time series of metrics
   - Distribution plots
   - Correlation analysis

### Configuration

```yaml
# webui/config.yaml
server:
  host: 0.0.0.0
  port: 5000
  debug: false

data:
  results_dir: ../evaluation_results
  max_upload_size: 100  # MB

visualization:
  default_metrics: [MRR, MAP, NDCG@5, Recall@5]
  color_scheme: seaborn
```

## Programmatic Visualization

### Using the Visualization API

```python
from evaluator.visualization import Visualizer

# Create visualizer
viz = Visualizer(results)

# Generate all plots
viz.create_summary_dashboard()
viz.save("output/dashboard.html")

# Individual plots
viz.plot_metrics_over_time()
viz.plot_error_breakdown()
viz.plot_model_comparison()
```

### Custom Plots

```python
import matplotlib.pyplot as plt
from evaluator.visualization import prepare_plot_data

# Prepare data
data = prepare_plot_data(results, metric="MRR")

# Custom plot
plt.figure(figsize=(10, 6))
plt.bar(data.labels, data.values)
plt.title("MRR by Experiment")
plt.ylabel("MRR")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mrr_comparison.png")
```

### Interactive Plots with Plotly

```python
import plotly.graph_objects as go
from evaluator.visualization import get_metric_series

# Get metric time series
mrr_series = get_metric_series(results, metric="MRR")

# Create interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=mrr_series.index,
    y=mrr_series.values,
    mode='lines+markers',
    name='MRR'
))

fig.update_layout(
    title="MRR Over Time",
    xaxis_title="Experiment",
    yaxis_title="MRR"
)

fig.show()
```

## Export Formats

### LaTeX Tables

```python
from evaluator.visualization import export_latex_table

# Export for publication
latex = export_latex_table(
    results,
    metrics=["MRR", "MAP", "NDCG@5", "Recall@5"],
    format="booktabs",  # Professional formatting
    caption="Evaluation Results",
    label="tab:results"
)

with open("results_table.tex", "w") as f:
    f.write(latex)
```

### Excel Spreadsheets

```python
from evaluator.visualization import export_excel

export_excel(
    results,
    filename="results.xlsx",
    include_plots=True,  # Embed plots in Excel
    include_stats=True   # Include statistical summary
)
```

### Publication-Ready Figures

```python
from evaluator.visualization import create_publication_figure

fig = create_publication_figure(
    results,
    figure_type="comparison",
    style="ieee",  # or "acm", "springer"
    dpi=300,
    format="pdf"
)

fig.save("figure1.pdf")
```

## Analysis Examples

### Ablation Study Visualization

```python
from evaluator.visualization import plot_ablation_study

# Assuming ablation results
ablation_results = {
    "Full System": results_full,
    "No ASR": results_no_asr,
    "No Embedding": results_no_embedding,
    "Baseline": results_baseline
}

plot_ablation_study(
    ablation_results,
    metric="MRR",
    show_confidence=True
)
```

### Error Analysis

```python
from evaluator.visualization import analyze_errors

# Identify problematic samples
error_analysis = analyze_errors(results)

# Plot error distribution
error_analysis.plot_distribution()

# Get top errors
top_errors = error_analysis.get_top_errors(n=10)
for error in top_errors:
    print(f"Sample {error.id}: {error.description}")
```

### Model Comparison Matrix

```python
from evaluator.visualization import plot_comparison_matrix

# Compare multiple models
models = {
    "Whisper-Tiny": results_tiny,
    "Whisper-Base": results_base,
    "Whisper-Medium": results_medium,
    "Whisper-Large": results_large
}

plot_comparison_matrix(
    models,
    metrics=["MRR", "MAP", "WER"],
    annotate=True,  # Show values
    cmap="RdYlGn"   # Red-Yellow-Green colormap
)
```

### Metric Correlation

```python
from evaluator.visualization import plot_metric_correlation

# See how metrics correlate
plot_metric_correlation(
    results,
    metrics=["MRR", "MAP", "NDCG@5", "Recall@5", "WER"]
)
```

## Best Practices

### 1. Save Raw Results

Always save complete results for later analysis:

```python
results.save("results/experiment_name.json")
results.to_dataframe().to_csv("results/experiment_name.csv")
```

### 2. Use Version Control

Track result versions:

```python
import json
from evaluator.core.result_version import add_version_to_result

result_dict = add_version_to_result(results.to_dict())
with open("experiment_v1.json", "w") as f:
    json.dump(result_dict, f, indent=2)
```

### 3. Document Experiments

Add metadata to results:

```python
results.metadata.update({
    "description": "Testing whisper-large with new corpus",
    "date": "2024-03-31",
    "author": "researcher_name",
    "notes": "Improved preprocessing pipeline"
})
```

### 4. Generate Reports

Create automated reports:

```python
from evaluator.visualization import generate_report

report = generate_report(
    results,
    template="standard",  # or "detailed", "summary"
    output_format="html"  # or "pdf", "markdown"
)

report.save("reports/experiment_report.html")
```

## Troubleshooting

### Jupyter Kernel Issues

```bash
# Install kernel
python -m ipykernel install --user --name evaluator

# Select kernel in Jupyter: Kernel → Change Kernel → evaluator
```

### Plotly Not Rendering

```bash
# Install plotly extensions
pip install jupyterlab-plotly
jupyter labextension install jupyterlab-plotly
```

### Large Result Files

For large result files, use chunked loading:

```python
from evaluator.visualization import ChunkedResultsLoader

loader = ChunkedResultsLoader(chunk_size=1000)
for chunk in loader.load_chunks("large_results.json"):
    process_chunk(chunk)
```

## Next Steps

- Explore notebooks in `notebooks/` directory
- See [Configuration](configuration.md) for saving visualization settings
- Check [WEBUI.md](../../WEBUI.md) for detailed Web UI documentation
