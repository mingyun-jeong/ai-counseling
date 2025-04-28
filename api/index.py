from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    text: str
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

class CounselingRequest(BaseModel):
    daily_note: str
    emotion: str  # "슬픔" or "기쁨"
    response_mode: str  # "감성적" or "이성적"
    max_length: int = 200

@app.get("/")
def read_root():
    return {"message": "AI Counseling Bot API is running"}

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "environment": "vercel"}

@app.post("/api/v1/chat")
def chat(req: ChatRequest):
    return {
        "reply": f"테스트 응답입니다. 입력: {req.text[:20]}..."
    }

@app.post("/api/v1/counseling")
def counseling(req: CounselingRequest):
    response = "오늘 하루도 수고하셨습니다. 좋은 하루 되세요!"
    summary = "테스트 요약입니다"
    
    return {
        "reply": response,
        "summary": summary
    } 