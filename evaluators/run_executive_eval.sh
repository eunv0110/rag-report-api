#!/bin/bash
cd /home/work/rag/Project/rag-report-generator
source .venv/bin/activate

echo "========================================"
echo "임원 보고서 평가 (Faithfulness 우선)"
echo "========================================"
echo ""

echo "=== 1/5: OpenAI + RRF + MultiQuery (Top-K: 6,8,10,12) ⭐⭐⭐ ==="
python evaluators/evaluate_report_types.py \
  --report-type executive \
  --retrievers openai_rrf_multiquery \
  --version v1

echo ""
echo "=== 2/5: OpenAI + RRF Ensemble (Top-K: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py \
  --report-type executive \
  --retrievers openai_rrf_ensemble \
  --version v1

echo ""
echo "=== 3/5: BGE-M3 + RRF + LC + Time (Top-K: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py \
  --report-type executive \
  --retrievers bge_m3_rrf_lc_time \
  --version v1

echo ""
echo "=== 4/5: Upstage + RRF + MultiQuery + LC (Top-K: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py \
  --report-type executive \
  --retrievers upstage_rrf_multiquery_lc \
  --version v1

echo ""
echo "=== 5/5: Qwen + RRF + MultiQuery (Top-K: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py \
  --report-type executive \
  --retrievers qwen_rrf_multiquery \
  --version v1

echo ""
echo "✅ 임원 보고서 평가 완료!"
echo "총 평가: 5개 리트리버 × 4개 Top-K = 20개 조합"
