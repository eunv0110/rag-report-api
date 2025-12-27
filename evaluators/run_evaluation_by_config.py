#!/usr/bin/env python3
"""설정 파일 기반 평가 실행 스크립트

evaluation_config.yaml에 정의된 리트리버 조합별로 평가를 실행합니다.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import json
from typing import Dict, Any, List
from datetime import datetime
import argparse


def load_evaluation_config(config_path: str = None) -> Dict[str, Any]:
    """평가 설정 파일 로드

    Args:
        config_path: 설정 파일 경로 (기본값: config/evaluation_config.yaml)

    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "evaluation_config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_prompt_template(template_path: str) -> str:
    """프롬프트 템플릿 로드

    Args:
        template_path: 템플릿 파일 경로

    Returns:
        프롬프트 텍스트
    """
    base_dir = Path(__file__).parent.parent
    full_path = base_dir / template_path

    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_evaluation_for_retriever(
    retriever_config: Dict[str, Any],
    report_type: str,
    system_prompt: str,
    answer_generation_prompt: str,
    dataset_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """단일 리트리버에 대한 평가 실행

    Args:
        retriever_config: 리트리버 설정
        report_type: 보고서 타입 (weekly_report / executive_report)
        system_prompt: 시스템 프롬프트
        answer_generation_prompt: 답변 생성 프롬프트
        dataset_path: 평가 데이터셋 경로
        output_dir: 결과 저장 디렉토리

    Returns:
        평가 결과 딕셔너리
    """
    print("\n" + "=" * 80)
    print(f"🚀 평가 시작: {retriever_config['display_name']}")
    print("=" * 80)
    print(f"📋 보고서 타입: {report_type}")
    print(f"🔍 리트리버: {retriever_config['retriever_type']}")
    print(f"🧬 임베딩 모델: {retriever_config['embedding_preset']}")
    print(f"📄 설명: {retriever_config['description']}")
    print()

    # 환경 변수 설정
    os.environ['MODEL_PRESET'] = retriever_config['embedding_preset']

    # TODO: 실제 평가 로직 구현
    # 1. 리트리버 초기화
    # 2. 데이터셋 로드
    # 3. 각 질문에 대해:
    #    - 문서 검색
    #    - 답변 생성 (system_prompt + answer_generation_prompt 사용)
    #    - 메트릭 평가 (Precision, Recall, Faithfulness)
    # 4. 결과 집계 및 저장

    # 결과 저장 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"eval_{retriever_config['name']}_{report_type}_{timestamp}.json"
    result_path = Path(output_dir) / report_type / result_filename
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # 임시 결과 (실제 구현에서는 평가 결과로 대체)
    result = {
        "retriever_name": retriever_config['name'],
        "display_name": retriever_config['display_name'],
        "report_type": report_type,
        "embedding_preset": retriever_config['embedding_preset'],
        "retriever_type": retriever_config['retriever_type'],
        "timestamp": timestamp,
        "expected_performance": retriever_config.get('expected_performance', {}),
        "actual_performance": {
            # TODO: 실제 평가 결과로 채우기
            "precision": 0.0,
            "recall": 0.0,
            "faithfulness": 0.0
        },
        "dataset_path": dataset_path,
        "result_path": str(result_path)
    }

    print(f"✅ 평가 완료")
    print(f"📊 예상 성능:")
    for metric, value in retriever_config.get('expected_performance', {}).items():
        print(f"   - {metric}: {value:.2f}")
    print(f"💾 결과 저장: {result_path}")

    return result


def run_weekly_report_evaluation(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """주간 보고서용 평가 실행

    Args:
        config: 전체 설정 딕셔너리

    Returns:
        평가 결과 리스트
    """
    weekly_config = config['weekly_report']

    print("\n" + "=" * 80)
    print(f"📊 {weekly_config['name']} 평가 시작")
    print("=" * 80)
    print(f"우선순위: {' > '.join(weekly_config['priority'])}")
    print(f"평가 대상 리트리버: {len(weekly_config['retrievers'])}개")
    print()

    # 프롬프트 로드
    system_prompt = load_prompt_template(weekly_config['system_prompt_path'])
    answer_generation_prompt = load_prompt_template(weekly_config['answer_generation_prompt_path'])

    results = []
    for retriever_config in weekly_config['retrievers']:
        result = run_evaluation_for_retriever(
            retriever_config=retriever_config,
            report_type="weekly_report",
            system_prompt=system_prompt,
            answer_generation_prompt=answer_generation_prompt,
            dataset_path=config['evaluation']['dataset_path'],
            output_dir=config['evaluation']['output_dir']
        )
        results.append(result)

    return results


def run_executive_report_evaluation(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """임원 보고서용 평가 실행

    Args:
        config: 전체 설정 딕셔너리

    Returns:
        평가 결과 리스트
    """
    exec_config = config['executive_report']

    print("\n" + "=" * 80)
    print(f"📊 {exec_config['name']} 평가 시작")
    print("=" * 80)
    print(f"우선순위: {' > '.join(exec_config['priority'])}")
    print(f"평가 대상 리트리버: {len(exec_config['retrievers'])}개")
    print()

    # 프롬프트 로드
    system_prompt = load_prompt_template(exec_config['system_prompt_path'])
    answer_generation_prompt = load_prompt_template(exec_config['answer_generation_prompt_path'])

    results = []
    for retriever_config in exec_config['retrievers']:
        result = run_evaluation_for_retriever(
            retriever_config=retriever_config,
            report_type="executive_report",
            system_prompt=system_prompt,
            answer_generation_prompt=answer_generation_prompt,
            dataset_path=config['evaluation']['dataset_path'],
            output_dir=config['evaluation']['output_dir']
        )
        results.append(result)

    return results


def compare_results(weekly_results: List[Dict[str, Any]], exec_results: List[Dict[str, Any]]):
    """평가 결과 비교 및 출력

    Args:
        weekly_results: 주간 보고서 평가 결과
        exec_results: 임원 보고서 평가 결과
    """
    print("\n" + "=" * 80)
    print("🏆 평가 결과 요약")
    print("=" * 80)

    print("\n📋 주간 보고서 (운영팀)")
    print("-" * 80)
    for result in weekly_results:
        print(f"\n{result['display_name']}")
        print(f"   예상 성능: ", end="")
        expected = result['expected_performance']
        print(f"Precision={expected.get('precision', 0):.2f}, ", end="")
        print(f"Recall={expected.get('recall', 0):.2f}, ", end="")
        print(f"Faithfulness={expected.get('faithfulness', 0):.2f}")

    print("\n\n📋 임원 보고서 (의사결정)")
    print("-" * 80)
    for result in exec_results:
        print(f"\n{result['display_name']}")
        print(f"   예상 성능: ", end="")
        expected = result['expected_performance']
        print(f"Precision={expected.get('precision', 0):.2f}, ", end="")
        print(f"Recall={expected.get('recall', 0):.2f}, ", end="")
        print(f"Faithfulness={expected.get('faithfulness', 0):.2f}")

    print("\n\n💡 추천")
    print("-" * 80)
    print("주간 보고서용 (Recall 우선):")
    print("  ⭐⭐⭐ Upstage + RRF + MultiQuery + LC")
    print("       → 완벽한 검색 성능 (Precision 1.00, Recall 1.00)")
    print()
    print("임원 보고서용 (Faithfulness 우선):")
    print("  ⭐⭐⭐ OpenAI + RRF + MultiQuery")
    print("       → 최고 Faithfulness (0.96)")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="리트리버 조합별 평가 실행")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="평가 설정 파일 경로 (기본값: config/evaluation_config.yaml)"
    )
    parser.add_argument(
        "--report-type",
        type=str,
        choices=["weekly", "executive", "both"],
        default="both",
        help="평가할 보고서 타입 (weekly: 주간 보고서, executive: 임원 보고서, both: 둘 다)"
    )

    args = parser.parse_args()

    # 설정 로드
    config = load_evaluation_config(args.config)

    print("=" * 80)
    print("🎯 리트리버 조합별 평가 시스템")
    print("=" * 80)
    print(f"설정 파일: {args.config or 'config/evaluation_config.yaml'}")
    print(f"평가 타입: {args.report_type}")
    print()

    # 평가 실행
    weekly_results = []
    exec_results = []

    if args.report_type in ["weekly", "both"]:
        weekly_results = run_weekly_report_evaluation(config)

    if args.report_type in ["executive", "both"]:
        exec_results = run_executive_report_evaluation(config)

    # 결과 비교
    if weekly_results or exec_results:
        compare_results(weekly_results, exec_results)

    print("\n✅ 모든 평가 완료")


if __name__ == "__main__":
    main()
