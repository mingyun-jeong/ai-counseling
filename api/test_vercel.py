from fastapi import FastAPI
from pydantic import BaseModel
import os

# 간단한 응답용 API
app = FastAPI(title="AI Counseling Bot Test API")

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
def root():
    return {"message": "AI Counseling Bot API is running"}

@app.get("/api/v1/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": "vercel", "deployment_test": "success"}

@app.post("/api/v1/chat")
def chat(req: ChatRequest):
    """Chat endpoint"""
    return {
        "reply": f"테스트 응답입니다. 입력: {req.text[:20]}...",
        "max_length": req.max_length,
        "temperature": req.temperature
    }

@app.post("/api/v1/counseling")
def counseling(req: CounselingRequest):
    """Counseling endpoint"""
    response = "오늘 하루도 수고하셨습니다. 좋은 하루 되세요!"
    summary = "테스트 요약입니다"
    
    return {
        "reply": response,
        "summary": summary
    } 