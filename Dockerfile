FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사하여 캐싱 최적화
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 모델 경로 및 환경 변수 설정
ENV MODEL_PATH=/app/models/1bit-llm
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# API 서버 시작 (첫 실행 시 자동으로 모델 다운로드)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 