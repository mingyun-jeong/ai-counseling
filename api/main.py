from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import logging
import re
from dotenv import load_dotenv
import random
from enum import Enum
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmotionType(str, Enum):
    SAD = "슬픔"
    HAPPY = "기쁨"

class ResponseMode(str, Enum):
    EMOTIONAL = "감성적"
    RATIONAL = "이성적"

class ChatRequest(BaseModel):
    text: str
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

class CounselingRequest(BaseModel):
    daily_note: str
    emotion: EmotionType
    response_mode: ResponseMode
    max_length: int = 200

app = FastAPI(title="AI Counseling Bot API", 
              description="한국어 상담 AI 챗봇 API - MS 1bit LLM 활용")

# Initialize model globals to None
model = None
tokenizer = None

# 태그 패턴: <|TAG|> 형식의 모든 태그
TAG_PATTERN = re.compile(r'<\|[A-Z_]+\|>|<\|[A-Z_]+ \|>|<\|END_[A-Z_]+\|>')

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    # Get model path from environment variable or use default
    model_path = os.getenv('MODEL_PATH', 'models/1bit-llm')
    
    logger.info(f"Loading model from {model_path}")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Try to load with quantization if CUDA is available
        if device == "cuda":
            try:
                import bitsandbytes as bnb
                logger.info("Using CUDA with bitsandbytes for quantized inference")
                
                # Create quantization config
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load model with quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            except Exception as e:
                logger.error(f"Error with quantization: {e}")
                logger.info("Falling back to standard CUDA loading")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        else:
            # CPU loading - no quantization
            logger.info("Using CPU for inference (no quantization)")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": device},
                torch_dtype=torch.float32  # Use float32 for CPU
            )
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def clean_response(text):
    """특수 태그와 메타 텍스트를 제거하고 응답을 정리합니다."""
    # 모든 <|TAG|> 형식의 태그 제거
    cleaned = TAG_PATTERN.sub('', text)
    
    # 여러 줄바꿈을 하나로 정리
    cleaned = re.sub(r'\n{2,}', '\n', cleaned)
    
    # 이상한 반복 패턴 제거 (예: '이제 이제 이제')
    cleaned = re.sub(r'(\S+)(\s+\1){2,}', r'\1', cleaned)
    
    # 응답의 첫 문장만 추출 (선택적)
    sentences = cleaned.split('\n')
    if sentences and sentences[0].strip():
        return sentences[0].strip()
    
    return cleaned.strip()

@app.post('/chat')
def chat(req: ChatRequest):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 사용자 입력 받기
        user_input = req.text
        logger.info(f"Received chat request: {user_input[:30]}...")
        
        # 사전 정의된 응답 목록
        predefined_responses = {
            # 우울, 슬픔, 무기력 관련
            "우울": [
                "우울한 감정을 느끼는 것은 자연스러운 일입니다. 언제부터 이런 감정을 느끼기 시작하셨나요?",
                "그런 감정이 들었군요. 우울함이 일상생활에 어떤 영향을 미치고 있나요?",
                "우울함은 많은 사람들이 경험하는 감정입니다. 특별히 우울함을 느끼게 하는 상황이 있었나요?"
            ],
            "슬픔": [
                "슬픔을 느끼시는군요. 그런 감정이 드는 특별한 계기가 있었나요?",
                "슬픔은 우리 삶의 일부입니다. 그 감정을 어떻게 대처하고 계신가요?",
                "그런 감정을 느끼실 때, 과거에는 어떻게 극복하셨나요?"
            ],
            "무기력": [
                "무기력함을 느끼시는군요. 일상에서 작은 성취감을 느끼는 활동을 시도해보는 것이 도움이 될 수 있어요.",
                "그런 상황이 힘드시겠네요. 무기력함을 줄이기 위해 시도해본 방법이 있나요?",
                "무기력함은 많은 사람들이 경험합니다. 언제부터 이런 감정이 시작되었나요?"
            ],
            
            # 직장, 스트레스 관련
            "직장": [
                "직장에서의 스트레스가 심하시군요. 구체적으로 어떤 부분이 가장 힘드신가요?",
                "직장 생활이 많이 힘드신 것 같네요. 혹시 특별히 스트레스를 주는 요소가 있나요?",
                "직장에서의 어려움은 많은 사람들이 경험합니다. 과거에는 이런 상황을 어떻게 대처하셨나요?"
            ],
            "스트레스": [
                "스트레스가 많이 쌓이셨군요. 평소에 스트레스를 푸는 방법이 있으신가요?",
                "스트레스를 느끼는 상황이 지속되면 힘드시겠네요. 특별히 어떤 상황에서 스트레스를 많이 느끼시나요?",
                "스트레스 관리는 매우 중요합니다. 혹시 이전에 도움이 되었던 스트레스 해소 방법이 있으신가요?"
            ],
            "일": [
                "일과 관련된 어려움이 있으신 것 같네요. 구체적으로 어떤 부분이 가장 부담되시나요?",
                "업무 상황이 힘드시군요. 어떤 종류의 업무가 특히 부담이 되시나요?",
                "일에서 오는 압박감이 크신 것 같습니다. 이런 상황에서 어떤 도움이 필요하신가요?"
            ],
            
            # 관계, 갈등 관련
            "갈등": [
                "인간관계에서 갈등은 자연스러운 일입니다. 어떤 상황에서 갈등이 발생하나요?",
                "관계에서의 갈등이 있으시군요. 그 상황에서 어떤 감정을 느끼셨나요?",
                "갈등 상황이 힘드시겠네요. 이전에는 비슷한 상황을 어떻게 해결하셨나요?"
            ],
            "관계": [
                "인간관계의 어려움은 누구에게나 큰 도전입니다. 특별히 어떤 관계가 힘드신가요?",
                "관계에서 어려움을 겪고 계시는군요. 어떤 부분이 가장 힘드신가요?",
                "관계 문제는 정말 어려울 수 있습니다. 그 관계에서 어떤 변화를 원하시나요?"
            ],
            "사람": [
                "다른 사람들과의 관계에서 어려움이 있으신 것 같네요. 어떤 상황이 가장 힘드신가요?",
                "사람들과의 상호작용이 힘드실 때가 있으시군요. 특별히 어떤 상황에서 그런 감정을 느끼시나요?",
                "사람관계에서 불편함을 느끼시는군요. 과거에는 이런 상황에서 어떻게 대처하셨나요?"
            ]
        }
        
        # 기본 응답
        default_responses = [
            "말씀해주신 내용에 공감합니다. 어떤 점이 가장 힘드신가요? 더 자세히 말씀해주시면 도움을 드리겠습니다.",
            "그런 상황이 정말 힘드셨겠네요. 지금 가장 필요한 것이 무엇인지 이야기해주실 수 있을까요?",
            "많이 힘드신 상황이네요. 혹시 이전에는 이런 상황을 어떻게 극복하셨나요?",
            "그런 감정을 느끼는 것은 매우 자연스러운 일입니다. 조금 더 구체적으로 말씀해주시겠어요?",
            "정말 어려운 상황이시네요. 지금 당장 작은 도움이 될 수 있는 것이 무엇이 있을까요?"
        ]
        
        # 키워드 매칭으로 응답 선택
        response = None
        for keyword, responses in predefined_responses.items():
            if keyword in user_input:
                response = random.choice(responses)
                logger.info(f"Matched keyword: {keyword}")
                break
        
        # 키워드가 없으면 기본 응답 사용
        if response is None:
            response = random.choice(default_responses)
            logger.info("Using default response")
        
        logger.info(f"Sending response: {response[:30]}...")
        return {'reply': response}
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/counseling')
def provide_counseling(req: CounselingRequest):
    """사용자의 하루 노트, 감정, 답변 모드에 따라 맞춤형 상담 응답을 제공합니다."""
    
    logger.info(f"Counseling request: note={req.daily_note[:30]}..., emotion={req.emotion}, mode={req.response_mode}")
    
    try:
        # 감정에 따른 응답 패턴
        emotional_responses = {
            EmotionType.SAD: {
                ResponseMode.EMOTIONAL: [
                    "그런 상황에서 슬픔을 느끼는 건 정말 자연스러운 일이에요. 팀장님의 말씀이 마음에 상처를 주었군요. 당신의 감정은 충분히 타당하고, 그런 기분이 들 수 있어요. 지금은 자신을 돌보는 시간을 가져보는 건 어떨까요?",
                    "슬픈 감정이 들어 많이 힘드시겠네요. 직장에서의 관계는 때로 우리 마음에 큰 영향을 미치죠. 당신의 마음이 아프다는 것을 인정하고, 스스로를 위로해주세요. 이런 감정을 느끼는 것은 당신이 섬세하고 진실된 사람이라는 증거이기도 해요.",
                    "팀장님의 말씀에 마음이 무거우시겠어요. 직장에서 상처받는 일은 정말 큰 슬픔을 줄 수 있어요. 당신의 감정을 있는 그대로 받아들이고, 오늘 하루 작은 행복을 찾아보는 건 어떨까요? 따뜻한 차 한 잔이나 좋아하는 음악을 들으며 마음을 달래보세요."
                ],
                ResponseMode.RATIONAL: [
                    "팀장님의 피드백이 있으셨군요. 이런 상황에서 슬픔을 느끼는 것은 자연스럽습니다. 하지만 이것을 성장의 기회로 삼을 수도 있습니다. 피드백의 내용을 객관적으로 분석하고, 개선할 점이 있다면 어떤 것인지 생각해보는 것이 도움이 될 수 있습니다.",
                    "직장에서의 피드백은 때로 감정적으로 받아들이기 어려울 수 있습니다. 슬픔을 느끼는 것은 정상적인 반응이지만, 이제 그 감정을 인식한 후에는 상황을 객관적으로 바라보는 것이 중요합니다. 팀장님의 의도와 맥락을 이해하려고 노력해보세요.",
                    "슬픈 감정이 들었을 때는 먼저 그 감정을 인정하고, 그다음 합리적인 대응 방안을 생각해보는 것이 좋습니다. 팀장님과의 소통에서 오해가 있었는지, 혹은 실제로 개선이 필요한 부분이 있는지 분석해보고 건설적인 해결책을 찾아보세요."
                ]
            },
            EmotionType.HAPPY: {
                ResponseMode.EMOTIONAL: [
                    "팀장님의 이야기에 기쁨을 느끼셨군요! 직장에서 인정받는 순간은 정말 행복한 경험이에요. 이런 긍정적인 감정을 충분히 누리고, 오늘 하루 이 기분을 유지하며 즐겁게 보내세요. 당신의 노력이 빛을 발하는 순간이네요!",
                    "와! 정말 기쁜 소식이네요. 팀장님과의 긍정적인 상호작용은 직장 생활에 큰 활력이 되죠. 이 기쁨을 마음껏 즐기시고, 이 긍정적인 에너지로 더 멋진 일들을 만들어가세요. 당신은 충분히 이런 행복한 순간을 누릴 자격이 있어요!",
                    "팀장님의 말씀에 행복을 느끼셨다니 정말 축하드려요! 우리 삶에서 이런 기쁜 순간들이 모여 큰 행복이 되죠. 이 기분을 오래 간직하시고, 앞으로도 이런 순간들이 더 많아지길 바랍니다. 당신의 성과와 노력이 인정받는 순간이네요."
                ],
                ResponseMode.RATIONAL: [
                    "팀장님과의 긍정적인 상호작용이 있으셨군요. 기쁨을 느끼는 것은 자연스러운 반응입니다. 이런 긍정적인 피드백은 앞으로의 업무 수행에 좋은 동기부여가 될 것입니다. 이 경험을 통해 어떤 부분이 좋은 평가를 받았는지 분석해보면 앞으로도 도움이 될 것입니다.",
                    "직장에서 긍정적인 피드백을 받아 기쁘다는 것은 전문적 성장의 중요한 지표입니다. 이런 순간을 통해 어떤 업무 방식이나 행동이 조직에서 가치 있게 평가되는지 파악할 수 있습니다. 앞으로도 이런 피드백을 지속적으로 받을 수 있는 전략을 고려해보세요.",
                    "팀장님의 긍정적인 말씀에 기쁨을 느끼셨다니 좋은 소식입니다. 이런 경험은 직업적 자신감과 만족도를 높이는 데 큰 역할을 합니다. 어떤 행동이나 성과가 이런 결과를 가져왔는지 객관적으로 분석하면, 앞으로의 경력 개발에 유용한 통찰을 얻을 수 있을 것입니다."
                ]
            }
        }
        
        # 응답 생성
        responses = emotional_responses[req.emotion][req.response_mode]
        full_response = random.choice(responses)
        
        # 요약 생성
        summary = generate_summary(req.daily_note, req.emotion, req.response_mode, full_response)
        
        return {
            "reply": full_response,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error generating counseling response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_summary(note, emotion, mode, response):
    """상담 응답에 대한 한 줄 요약을 생성합니다."""
    
    summary_templates = {
        EmotionType.SAD: {
            ResponseMode.EMOTIONAL: [
                "팀장님의 말씀에 느낀 슬픔을 공감하며 자신을 돌보는 시간을 가질 것을 제안합니다.",
                "직장에서 받은 상처와 슬픔을 인정하고 스스로를 위로하도록 격려합니다.",
                "슬픈 감정을 있는 그대로 받아들이고 작은 위안을 찾도록 조언합니다."
            ],
            ResponseMode.RATIONAL: [
                "슬픔을 인정하면서도 이를 성장의 기회로 삼아 객관적으로 상황을 분석할 것을 제안합니다.",
                "감정을 인식한 후 상황을 객관적으로 바라보고 팀장님의 의도를 이해하도록 조언합니다.",
                "슬픈 감정을 인정하고 건설적인 해결책을 찾기 위한 분석을 권장합니다."
            ]
        },
        EmotionType.HAPPY: {
            ResponseMode.EMOTIONAL: [
                "팀장님의 긍정적인 말씀에서 비롯된 기쁨을 충분히 누리고 즐기도록 격려합니다.",
                "직장에서의 긍정적 상호작용에서 오는 행복을 만끽하고 이 에너지를 유지하도록 응원합니다.",
                "인정받는 순간의 기쁨을 오래 간직하고 노력이 빛을 발하는 순간을 축하합니다."
            ],
            ResponseMode.RATIONAL: [
                "긍정적 피드백을 업무 동기부여의 기회로 삼고 잘된 부분을 분석할 것을 제안합니다.",
                "기쁜 감정을 통해 조직에서 가치 있게 평가되는 행동을 파악하도록 조언합니다.",
                "긍정적 경험을 직업적 자신감으로 연결하고 성공 요인을 객관적으로 분석할 것을 권장합니다."
            ]
        }
    }
    
    summaries = summary_templates[emotion][mode]
    return random.choice(summaries)

@app.get('/health')
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None} 