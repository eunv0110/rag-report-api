#!/usr/bin/env python3
"""Langfuse REST API로 평가 결과를 가져와 리트리버 비교

사용법:
    # 모든 리트리버 비교
    python scripts/compare_retrievers_from_langfuse.py

    # 특정 보고서 타입만
    python scripts/compare_retrievers_from_langfuse.py --report-type weekly_report

    # 특정 메트릭으로 정렬
    python scripts/compare_retrievers_from_langfuse.py --sort-by context_recall
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
from collections import defaultdict
import base64

from config.settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST


def get_auth_header() -> Dict[str, str]:
    """Langfuse API 인증 헤더 생성"""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        raise ValueError("Langfuse 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

    # Basic Auth 생성
    credentials = f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}"
    encoded = base64.b64encode(credentials.encode()).decode()

    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json"
    }


def fetch_trace_by_id(trace_id: str) -> Optional[Dict[str, Any]]:
    """특정 trace ID로 trace 정보 가져오기"""
    base_url = LANGFUSE_HOST.rstrip('/')
    url = f"{base_url}/api/public/traces/{trace_id}"

    headers = get_auth_header()

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except:
        return None


def fetch_all_scores(max_pages: int = 50) -> List[Dict[str, Any]]:
    """모든 scores 가져오기"""
    base_url = LANGFUSE_HOST.rstrip('/')
    url = f"{base_url}/api/public/scores"
    headers = get_auth_header()

    all_scores = []

    print("\n" + "=" * 80)
    print("🔍 Langfuse Scores 가져오는 중...")
    print("=" * 80)

    for page in range(1, max_pages + 1):
        params = {
            "page": page,
            "limit": 100
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            scores = data.get("data", [])

            if not scores:
                print(f"   Page {page}: 데이터 없음 - 중단")
                break

            all_scores.extend(scores)
            print(f"   Page {page}: {len(scores)} scores 발견 (누적: {len(all_scores)})")

        except requests.exceptions.HTTPError as e:
            print(f"  ❌ HTTP 에러: {e}")
            print(f"     Response: {e.response.text}")
            break
        except Exception as e:
            print(f"  ❌ API 호출 실패: {e}")
            break

    print(f"✅ 총 {len(all_scores)} scores 발견")
    return all_scores


def aggregate_scores_by_retriever(
    all_scores: List[Dict[str, Any]],
    report_type_filter: Optional[str] = None
) -> Dict[str, Dict[str, List[float]]]:
    """Scores를 retriever별로 집계"""

    print("\n📊 Scores 집계 중...")

    retriever_scores = defaultdict(lambda: defaultdict(list))
    trace_cache = {}  # trace 정보 캐싱

    processed = 0
    skipped = 0

    for score in all_scores:
        trace_id = score.get("traceId")
        score_name = score.get("name", "").lower()
        score_value = score.get("value")

        # 관심있는 메트릭만
        if score_name not in ["faithfulness", "context_precision", "context_recall"]:
            continue

        if score_value is None:
            continue

        # Trace 정보 가져오기 (캐싱)
        if trace_id not in trace_cache:
            trace = fetch_trace_by_id(trace_id)
            if trace:
                trace_cache[trace_id] = trace
            else:
                trace_cache[trace_id] = None

        trace = trace_cache[trace_id]
        if not trace:
            skipped += 1
            continue

        # Metadata에서 retriever 정보 추출
        metadata = trace.get("metadata", {})
        retriever_name = metadata.get("retriever_name")
        trace_report_type = metadata.get("report_type")
        top_k = metadata.get("top_k", "unknown")

        # Report type 필터링
        if report_type_filter and trace_report_type != report_type_filter:
            skipped += 1
            continue

        if not retriever_name:
            skipped += 1
            continue

        # Key 생성: retriever_name + top_k
        key = f"{retriever_name}_k{top_k}"
        retriever_scores[key][score_name].append(score_value)
        processed += 1

        if processed % 100 == 0:
            print(f"   ✓ {processed} scores 처리됨 (스킵: {skipped})...")

    print(f"✅ 총 {processed} scores 처리 완료 (스킵: {skipped})")
    print(f"✅ {len(retriever_scores)} 개 리트리버 발견")

    return dict(retriever_scores)


def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """점수 통계 계산"""
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "count": len(scores)
    }


def create_comparison_dataframe(retriever_scores: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """비교 테이블 생성"""
    rows = []

    for retriever_name, scores_dict in retriever_scores.items():
        row = {"retriever": retriever_name}

        # 각 메트릭의 평균 계산
        for metric in ["faithfulness", "context_precision", "context_recall"]:
            if metric in scores_dict and scores_dict[metric]:
                stats = calculate_statistics(scores_dict[metric])
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_count"] = stats["count"]
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_count"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame, sort_by: str = "faithfulness_mean"):
    """비교 테이블 출력"""
    if df.empty:
        print("❌ 비교할 데이터가 없습니다.")
        return

    # 정렬
    if sort_by + "_mean" in df.columns:
        df = df.sort_values(sort_by + "_mean", ascending=False, na_position='last')

    print("\n" + "=" * 120)
    print("📊 리트리버 성능 비교")
    print("=" * 120)

    # 출력 포맷 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    # 주요 메트릭만 출력
    display_cols = ["retriever"]
    metric_cols = []

    for metric in ["faithfulness", "context_precision", "context_recall"]:
        mean_col = f"{metric}_mean"
        count_col = f"{metric}_count"
        if mean_col in df.columns and df[mean_col].notna().any():
            display_cols.append(mean_col)
            display_cols.append(count_col)
            metric_cols.append(mean_col)

    # 테이블 출력
    display_df = df[display_cols].copy()

    # 숫자 포맷팅
    for col in metric_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))
    print("=" * 120)

    # 최고 성능 리트리버
    print("\n🏆 최고 성능 리트리버:")
    for metric in ["faithfulness", "context_precision", "context_recall"]:
        mean_col = f"{metric}_mean"
        if mean_col in df.columns and df[mean_col].notna().any():
            # 원본 데이터프레임에서 최댓값 찾기
            numeric_df = df[df[mean_col].notna()].copy()
            if not numeric_df.empty:
                best_idx = numeric_df[mean_col].idxmax()
                best_retriever = numeric_df.loc[best_idx, "retriever"]
                best_score = numeric_df.loc[best_idx, mean_col]
                count = numeric_df.loc[best_idx, f"{metric}_count"]
                print(f"  • {metric.capitalize()}: {best_retriever} ({best_score:.3f}, n={int(count)})")


def export_to_csv(df: pd.DataFrame, output_path: str):
    """결과를 CSV로 저장"""
    df.to_csv(output_path, index=False)
    print(f"\n💾 결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Langfuse에서 리트리버 평가 결과 비교")
    parser.add_argument(
        "--report-type",
        type=str,
        choices=["weekly_report", "executive_report"],
        help="보고서 타입 필터"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="faithfulness",
        choices=["faithfulness", "context_precision", "context_recall"],
        help="정렬 기준 메트릭"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="가져올 최대 페이지 수 (페이지당 100개 scores)"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="결과를 CSV로 저장할 경로"
    )

    args = parser.parse_args()

    # Scores 가져오기
    try:
        all_scores = fetch_all_scores(max_pages=args.max_pages)
    except ValueError as e:
        print(f"❌ {e}")
        return

    if not all_scores:
        print("\n❌ Scores를 찾을 수 없습니다.")
        print("💡 힌트:")
        print("   1. Langfuse에 평가 결과가 업로드되었는지 확인")
        print("   2. .env 파일에 LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY가 설정되어 있는지 확인")
        return

    # Retriever별로 집계
    retriever_scores = aggregate_scores_by_retriever(
        all_scores,
        report_type_filter=args.report_type
    )

    if not retriever_scores:
        print("\n❌ 유효한 평가 결과를 찾을 수 없습니다.")
        print("💡 힌트:")
        print("   1. trace의 metadata에 'retriever_name'이 포함되어 있는지 확인")
        print("   2. report_type 필터가 올바른지 확인")
        return

    # 비교 테이블 생성
    df = create_comparison_dataframe(retriever_scores)

    # 결과 출력
    print_comparison_table(df, sort_by=args.sort_by)

    # CSV 저장
    if args.export:
        export_to_csv(df, args.export)


if __name__ == "__main__":
    main()
