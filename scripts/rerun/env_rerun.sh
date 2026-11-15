# Source before `evaluator run`. Paths for the 2026-11-15 re-runs (per-query stats).
source ~/miniconda/etc/profile.d/conda.sh
conda activate ai-env

export CUDA_VISIBLE_DEVICES=3,4,5          # GPUs 0-2 busy with other jobs; configs' cuda:0/1 map to 3/4
export APM_CHECKPOINT_DIR=~/dump/apm/models
export EAM_CHECKPOINT_DIR=~/dump/eam/eam_alignment   # symlinks to runs/runs/model_base_final
export ADMED_WHISPER_LORA_PATH=~/models/medical-transcribe-pl/natural_anonym_synth_cv_fl
export ADMED_VOICE_PATH=~/.cache/kagglehub/datasets/karmarci/polish-medical-multi-source-corpus/versions/9

PQA=~/datasets/pubmedqa
# PubMedQA paths differ per config (same as August):
#   EN 1000 runs:  PUBMED_QA_DATASET=$PQA/pubmed_qa_1000_labeled
#   *_pl runs:     PUBMED_QA_DATASET_PL=$PQA/pubmed_qa_100_labeled_abstract_pl
#   *_plen runs:   PUBMED_QA_DATASET_PL=... (as above) + PUBMED_QA_DATASET=$PQA/pubmed_qa_100_labeled_abstract
pqa_en()   { export PUBMED_QA_DATASET=$PQA/pubmed_qa_1000_labeled; }
pqa_pl()   { export PUBMED_QA_DATASET_PL=$PQA/pubmed_qa_100_labeled_abstract_pl; }
pqa_plen() { pqa_pl; export PUBMED_QA_DATASET=$PQA/pubmed_qa_100_labeled_abstract; }
