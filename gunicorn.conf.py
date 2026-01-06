"""Gunicorn 설정 파일 - RAG Report Generator API"""

import multiprocessing
import os

# 서버 소켓
bind = f"0.0.0.0:{os.getenv('API_PORT', '8000')}"
backlog = 2048

# 워커 프로세스
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# 로깅
accesslog = "/home/work/rag/Project/rag-report-generator/logs/gunicorn-access.log"
errorlog = "/home/work/rag/Project/rag-report-generator/logs/gunicorn-error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 프로세스 이름
proc_name = "rag-report-api"

# 재시작
max_requests = 1000
max_requests_jitter = 50

# 보안
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# 개발 모드
reload = os.getenv('API_RELOAD', 'false').lower() == 'true'
