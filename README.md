# Korean AI Counseling Bot

한국어 상담 AI 챗봇 서비스 - 감정 공감과 심리 상담 조언을 제공하는 LLM 기반 상담 시스템

## 프로젝트 개요

이 프로젝트는 한국어 감정 공감 및 심리 상담을 제공하는 AI 챗봇을 구현합니다. Llama 2 모델을 LoRA 기법으로 파인튜닝하여 상담 맥락에 특화된 대화 모델을 학습합니다.

## 주요 기능

- 감정 공감 (Empathy): 사용자의 감정을 이해하고 공감적 반응 제공
- 심리 상담 (Counseling): 사용자의 심리적 어려움에 전문적인 조언 제공
- 경량 학습: LoRA를 통한 효율적인 모델 적응
- REST API: FastAPI 기반 상담 서비스 제공

## 시작하기

### 필수 요구사항

- Python 3.8+
- CUDA 지원 GPU (학습 및 추론용)
- Hugging Face 계정 및 Llama 2 모델 접근 권한

### 설치

```bash
git clone https://github.com/your-username/korean-counseling-bot.git
cd korean-counseling-bot
pip install -r requirements.txt
```

### 데이터 전처리

```bash
python scripts/preprocess.py data/raw.jsonl data/train.jsonl
```

### 모델 파인튜닝

```bash
python scripts/finetune.py
```

### API 서버 실행

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker 컨테이너 빌드 및 실행

```bash
docker build -t korean-counseling-bot .
docker run -p 8000:8000 --gpus all korean-counseling-bot
```

## API 사용법

### 상담 요청

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"text": "요즘 너무 우울해서 아무것도 하기 싫어요."}'
```

### 응답 예시

```json
{
  "reply": "그런 감정을 느끼고 계시는군요. 우울함은 누구나 경험할 수 있는 자연스러운 감정입니다. 하루에 한 가지 작은 일부터 시작해보는 건 어떨까요? 가벼운 산책이나 좋아하는 음악 듣기 같은 작은 활동이 도움이 될 수 있습니다."
}
```

## 프로젝트 구조

```
./
├── data              # 전처리 및 학습용 데이터
├── models            # 학습된 모델 및 LoRA weight 저장
├── scripts           # 데이터 전처리 및 fine-tuning 스크립트
│   ├── preprocess.py
│   └── finetune.py
├── api               # FastAPI 서버 코드
│   └── main.py
├── requirements.txt  # Python 패키지 목록
├── Dockerfile        # 컨테이너 이미지 빌드 설정
└── README.md         # 프로젝트 개요 및 사용법
``` 