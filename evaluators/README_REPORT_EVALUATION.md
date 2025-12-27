# 보고서 타입별 리트리버 평가 시스템

주간 보고서(운영팀)와 임원 보고서(의사결정) 각각에 최적화된 리트리버 조합을 평가하는 시스템입니다.

## 📋 목차

- [개요](#개요)
- [평가 전략](#평가-전략)
- [프롬프트 구조](#프롬프트-구조)
- [설정 파일](#설정-파일)
- [사용법](#사용법)
- [결과 분석](#결과-분석)

## 개요

### 주간 보고서 (운영팀)
- **우선순위**: Recall > Precision > Faithfulness
- **목적**: 놓치는 정보 없이 모든 내용 커버
- **사용자**: 운영팀 (일일 업무 수행)

### 임원 보고서 (의사결정)
- **우선순위**: Faithfulness > Precision > Recall
- **목적**: 틀린 정보 절대 안됨
- **사용자**: 임원진 (중요 의사결정)

## 평가 전략

### 주간 보고서용 리트리버

#### 1. Upstage + RRF + MultiQuery + LC ⭐⭐⭐ (추천)
- **임베딩**: Upstage Solar Embedding
- **리트리버**: RRF + MultiQuery + LongContext
- **예상 성능**: Precision 1.00, Recall 1.00, Faithfulness 0.76
- **장점**: 완벽한 검색 성능 + 준수한 Faithfulness

#### 2. Qwen + RRF Ensemble
- **임베딩**: Qwen 2.5 72B
- **리트리버**: RRF Ensemble
- **예상 성능**: Precision 0.96, Recall 0.99, Faithfulness 0.76
- **장점**: 안정적 백업 옵션

### 임원 보고서용 리트리버

#### 1. OpenAI + RRF + MultiQuery ⭐⭐⭐ (추천)
- **임베딩**: OpenAI text-embedding-3-large
- **리트리버**: RRF + MultiQuery
- **예상 성능**: Precision 0.95, Recall 0.87, Faithfulness 0.96
- **장점**: 전체 최고 Faithfulness, 임원 보고서에 가장 적합

#### 2. BGE-M3 + RRF + LC + Time
- **임베딩**: BGE-M3 (Jina Embeddings v3)
- **리트리버**: RRF + LongContext + TimeWeighted
- **예상 성능**: Precision 0.97, Recall 0.97, Faithfulness 0.80
- **장점**: 최신 정보 반영 + 높은 Faithfulness

## 프롬프트 구조

### 주간 보고서 프롬프트

**시스템 프롬프트** (`prompts/templates/evaluation/weekly_report/system_prompt.txt`):
- 완전성, 포괄성, 정확성 강조
- 모든 정보를 빠짐없이 포함하도록 지시
- 검색 결과에 없는 내용 추가 금지

**답변 생성 프롬프트** (`prompts/templates/evaluation/weekly_report/answer_generation_prompt.txt`):
```
## 주간 운영 보고서
### 1. 주요 지표
### 2. 주요 활동 및 진행사항
### 3. 이슈 및 문제점
### 4. 다음 주 예정사항
```

### 임원 보고서 프롬프트

**시스템 프롬프트** (`prompts/templates/evaluation/executive_report/system_prompt.txt`):
- 정확성, 신뢰성, 명확성 최우선
- 추측, 추론, 가정 절대 금지
- 불확실한 정보는 제외

**답변 생성 프롬프트** (`prompts/templates/evaluation/executive_report/answer_generation_prompt.txt`):
```
## 임원 보고서
### Executive Summary
### 1. 주요 현황
### 2. 핵심 이슈 및 리스크
### 3. 권고사항 (선택적)
### 4. 추가 확인 필요 사항
```

## 설정 파일

### `config/evaluation_config.yaml`

설정 파일에서 다음을 지정할 수 있습니다:

```yaml
weekly_report:
  retrievers:
    - name: "upstage_rrf_multiquery_lc"
      display_name: "Upstage + RRF + MultiQuery + LC"
      embedding_preset: "upstage"
      retriever_type: "rrf_multiquery_longcontext"
      k: 10  # Top-K 값 (개별 설정 가능)
      expected_performance:
        precision: 1.00
        recall: 1.00
        faithfulness: 0.76
```

**주요 설정 옵션**:
- `name`: 리트리버 고유 이름
- `embedding_preset`: 임베딩 모델 (upstage, qwen, openai, bge_m3)
- `retriever_type`: 리트리버 타입 (rrf_multiquery_longcontext, rrf_ensemble, 등)
- `k`: Top-K 값 (옵션, 미지정 시 커맨드라인 `--top-k` 값 사용)

## 사용법

### 1. 설정 확인

```bash
# 프롬프트 파일과 설정이 올바른지 테스트
python scripts/test_evaluation_setup.py
```

### 2. 평가 실행

#### 모든 평가 실행 (주간 + 임원)
```bash
python evaluators/evaluate_report_types.py --report-type both
```

#### 주간 보고서만 평가
```bash
python evaluators/evaluate_report_types.py --report-type weekly
```

#### 임원 보고서만 평가
```bash
python evaluators/evaluate_report_types.py --report-type executive
```

#### 특정 리트리버만 평가
```bash
python evaluators/evaluate_report_types.py \
  --report-type weekly \
  --retrievers upstage_rrf_multiquery_lc
```

#### Top-K 값 변경
```bash
python evaluators/evaluate_report_types.py \
  --report-type both \
  --top-k 15
```

**참고**: 개별 리트리버의 `k` 값은 설정 파일에서 지정 가능하며, 설정된 값이 커맨드라인 `--top-k`보다 우선됩니다.

#### 커스텀 데이터셋 사용
```bash
python evaluators/evaluate_report_types.py \
  --dataset data/evaluation/custom_qa_dataset.json \
  --report-type both
```

#### 버전 태그 지정
```bash
python evaluators/evaluate_report_types.py \
  --report-type both \
  --version v2
```

### 3. 결과 확인

#### Langfuse 대시보드
평가 결과는 Langfuse에 자동으로 기록됩니다:
- URL: https://cloud.langfuse.com
- 각 평가는 trace로 기록되며 다음 태그로 필터링 가능:
  - `weekly_report` / `executive_report`
  - 리트리버 이름 (예: `upstage_rrf_multiquery_lc_v1`)
  - 임베딩 프리셋 (예: `upstage`, `openai`)

#### 로컬 결과 파일
```
data/langfuse/evaluation_results/
├── weekly_report/
│   ├── upstage_rrf_multiquery_lc_stats.json
│   └── qwen_rrf_ensemble_stats.json
└── executive_report/
    ├── openai_rrf_multiquery_stats.json
    └── bge_m3_rrf_lc_time_stats.json
```

각 파일에는 다음 정보가 포함됩니다:
- 총 쿼리 수
- 평균 응답 시간
- 평균 컨텍스트 수
- 리트리버 설정 정보

## 결과 분석

### 메트릭 설명

1. **Precision (정밀도)**
   - 검색된 문서가 실제로 관련 있는지 측정
   - 높을수록 노이즈가 적음

2. **Recall (재현율)**
   - 관련된 모든 문서를 찾았는지 측정
   - 높을수록 정보 누락이 적음

3. **Faithfulness (충실도)**
   - 생성된 답변이 검색된 컨텍스트에 충실한지 측정
   - 높을수록 hallucination이 적음

### 평가 기준

**주간 보고서**:
- ✅ 모든 관련 정보가 포함되었는가? (Recall 최우선)
- ✅ 포함된 정보가 모두 정확한가? (Precision)
- ✅ 검색 결과를 충실히 반영했는가? (Faithfulness)

**임원 보고서**:
- ✅ 틀린 정보가 전혀 없는가? (Faithfulness 최우선)
- ✅ 불확실한 추측이 포함되지 않았는가? (Precision)
- ✅ 핵심 정보가 누락되지 않았는가? (Recall)

## 추가 정보

### 환경 변수 설정

평가 실행 전 `.env` 파일에 다음 설정이 필요합니다:

```bash
# Azure OpenAI (답변 생성용)
AZURE_AI_CREDENTIAL=your_credential
AZURE_AI_ENDPOINT=https://models.inference.ai.azure.com

# Langfuse (평가 추적용)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# 임베딩 모델별 API 키
UPSTAGE_API_KEY=your_key  # Upstage 사용 시
OPENAI_API_KEY=your_key    # OpenAI 사용 시
OPENROUTER_API_KEY=your_key  # Qwen 사용 시
```

### 문제 해결

**임베딩 캐시 초기화**:
```bash
rm -rf data/embeddings_cache/
```

**Langfuse 연결 확인**:
```bash
python -c "from utils.langfuse_utils import get_langfuse_client; print('✅' if get_langfuse_client() else '❌')"
```

### 관련 파일

- **평가 스크립트**: `evaluators/evaluate_report_types.py`
- **설정 파일**: `config/evaluation_config.yaml`
- **프롬프트**: `prompts/templates/evaluation/`
- **비교 스크립트**: `evaluators/compare_evaluation_results.py`
- **테스트**: `scripts/test_evaluation_setup.py`
