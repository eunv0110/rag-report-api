#!/usr/bin/env python3
"""평가 실행 헬퍼 모듈

리트리버와 프롬프트 조합별로 실제 평가를 수행합니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context


class EvaluationRunner:
    """평가 실행 클래스"""

    def __init__(
        self,
        embedding_preset: str,
        retriever_type: str,
        system_prompt: str,
        answer_generation_prompt: str,
        report_type: str
    ):
        """
        Args:
            embedding_preset: 임베딩 모델 프리셋 (upstage, qwen, openai, bge_m3)
            retriever_type: 리트리버 타입
            system_prompt: 시스템 프롬프트
            answer_generation_prompt: 답변 생성 프롬프트
            report_type: 보고서 타입 (weekly_report, executive_report)
        """
        self.embedding_preset = embedding_preset
        self.retriever_type = retriever_type
        self.system_prompt = system_prompt
        self.answer_generation_prompt = answer_generation_prompt
        self.report_type = report_type

        # 환경 변수 설정
        os.environ['MODEL_PRESET'] = embedding_preset

        # Langfuse 초기화
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

        # 컴포넌트 초기화는 지연 로딩 (필요 시 구현)
        self.retriever = None
        self.llm = None

    def initialize_retriever(self):
        """리트리버 초기화

        TODO: 실제 리트리버 타입에 따라 초기화 로직 구현
        """
        # from retrievers import get_retriever
        # self.retriever = get_retriever(self.retriever_type)
        pass

    def initialize_llm(self):
        """LLM 초기화

        TODO: 실제 LLM 초기화 로직 구현
        """
        # from langchain_openai import ChatOpenAI
        # self.llm = ChatOpenAI(...)
        pass

    @observe()
    def retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """문서 검색

        Args:
            question: 질문

        Returns:
            검색된 문서 리스트
        """
        if self.retriever is None:
            self.initialize_retriever()

        # TODO: 실제 검색 로직 구현
        # documents = self.retriever.get_relevant_documents(question)
        # return documents

        # 임시 반환
        return []

    @observe()
    def generate_answer(self, question: str, context: str) -> str:
        """답변 생성

        Args:
            question: 질문
            context: 검색된 문서 컨텍스트

        Returns:
            생성된 답변
        """
        if self.llm is None:
            self.initialize_llm()

        # 프롬프트 구성
        prompt = self.answer_generation_prompt.format(
            question=question,
            context=context
        )

        # TODO: 실제 LLM 호출 로직 구현
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": prompt}
        # ]
        # response = self.llm.invoke(messages)
        # return response.content

        # 임시 반환
        return ""

    def evaluate_single_qa(
        self,
        question: str,
        ground_truth_answer: str,
        ground_truth_contexts: List[str]
    ) -> Dict[str, float]:
        """단일 Q&A에 대한 평가

        Args:
            question: 질문
            ground_truth_answer: 정답
            ground_truth_contexts: 정답 컨텍스트

        Returns:
            평가 메트릭 딕셔너리 (precision, recall, faithfulness)
        """
        # 1. 문서 검색
        retrieved_docs = self.retrieve_documents(question)

        # 2. 컨텍스트 구성
        context = "\n\n".join([doc.get("content", "") for doc in retrieved_docs])

        # 3. 답변 생성
        generated_answer = self.generate_answer(question, context)

        # 4. 평가 메트릭 계산
        # TODO: 실제 평가 로직 구현
        # - Precision: 검색된 문서의 관련성
        # - Recall: 관련 문서를 얼마나 찾았는가
        # - Faithfulness: 생성된 답변이 컨텍스트에 충실한가

        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "faithfulness": 0.0
        }

        # Langfuse에 기록
        langfuse_context.update_current_trace(
            name=f"evaluation_{self.report_type}",
            metadata={
                "embedding_preset": self.embedding_preset,
                "retriever_type": self.retriever_type,
                "report_type": self.report_type
            }
        )

        langfuse_context.score_current_trace(
            name="precision",
            value=metrics["precision"]
        )

        langfuse_context.score_current_trace(
            name="recall",
            value=metrics["recall"]
        )

        langfuse_context.score_current_trace(
            name="faithfulness",
            value=metrics["faithfulness"]
        )

        return metrics

    def run_evaluation(self, dataset_path: str) -> Dict[str, Any]:
        """전체 데이터셋에 대한 평가 실행

        Args:
            dataset_path: 평가 데이터셋 경로

        Returns:
            평가 결과 딕셔너리
        """
        # 데이터셋 로드
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"📊 데이터셋 로드: {len(dataset)} 개 샘플")

        all_metrics = {
            "precision": [],
            "recall": [],
            "faithfulness": []
        }

        # 각 Q&A에 대해 평가
        for i, qa_pair in enumerate(dataset, 1):
            print(f"\n[{i}/{len(dataset)}] 평가 중...", end=" ")

            metrics = self.evaluate_single_qa(
                question=qa_pair.get("question", ""),
                ground_truth_answer=qa_pair.get("answer", ""),
                ground_truth_contexts=qa_pair.get("contexts", [])
            )

            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)

            print(f"✓ (P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F={metrics['faithfulness']:.2f})")

        # 평균 계산
        avg_metrics = {
            metric_name: sum(values) / len(values) if values else 0.0
            for metric_name, values in all_metrics.items()
        }

        result = {
            "embedding_preset": self.embedding_preset,
            "retriever_type": self.retriever_type,
            "report_type": self.report_type,
            "total_samples": len(dataset),
            "avg_metrics": avg_metrics,
            "all_metrics": all_metrics,
            "timestamp": datetime.now().isoformat()
        }

        return result


def create_evaluation_runner(
    embedding_preset: str,
    retriever_type: str,
    system_prompt_path: str,
    answer_generation_prompt_path: str,
    report_type: str
) -> EvaluationRunner:
    """평가 러너 생성 헬퍼 함수

    Args:
        embedding_preset: 임베딩 모델 프리셋
        retriever_type: 리트리버 타입
        system_prompt_path: 시스템 프롬프트 파일 경로
        answer_generation_prompt_path: 답변 생성 프롬프트 파일 경로
        report_type: 보고서 타입

    Returns:
        EvaluationRunner 인스턴스
    """
    base_dir = Path(__file__).parent.parent

    # 프롬프트 로드
    with open(base_dir / system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    with open(base_dir / answer_generation_prompt_path, 'r', encoding='utf-8') as f:
        answer_generation_prompt = f.read().strip()

    return EvaluationRunner(
        embedding_preset=embedding_preset,
        retriever_type=retriever_type,
        system_prompt=system_prompt,
        answer_generation_prompt=answer_generation_prompt,
        report_type=report_type
    )
