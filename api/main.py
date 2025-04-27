from fastapi import FastAPI, HTTPException, APIRouter
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

# API 버전 관리를 위한 라우터 생성
v1_router = APIRouter(prefix="/api/v1")

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

@v1_router.post('/chat')
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

@v1_router.post('/counseling')
def provide_counseling(req: CounselingRequest):
    """사용자의 하루 노트, 감정, 답변 모드에 따라 맞춤형 상담 응답을 제공합니다."""
    
    logger.info(f"Counseling request: note={req.daily_note[:30]}..., emotion={req.emotion}, mode={req.response_mode}")
    
    try:
        # 키워드 추출 - 노트 내용에서 중요 키워드 파악
        note_keywords = extract_keywords(req.daily_note)
        logger.info(f"Extracted keywords: {note_keywords}")
        
        # 노트 내용 분석
        note_analysis = analyze_note_content(req.daily_note)
        logger.info(f"Note analysis: {note_analysis}")
        
        # 맞춤형 응답 생성
        response = generate_custom_response(
            req.daily_note, 
            req.emotion, 
            req.response_mode, 
            note_keywords,
            note_analysis
        )
        
        # 요약 생성
        summary = generate_custom_summary(req.daily_note, req.emotion, req.response_mode, response, note_analysis)
        
        return {
            "reply": response,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error generating counseling response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_keywords(text):
    """텍스트에서 중요 키워드를 추출합니다."""
    # 간단한 키워드 추출 로직
    common_keywords = [
        "업무", "일", "직장", "회사", "프로젝트", "스트레스", "피로", 
        "친구", "관계", "가족", "갈등", "대화", "오해", "말다툼",
        "성공", "성취", "완료", "칭찬", "인정", "보상", "기회"
    ]
    
    # 텍스트를 소문자로 변환하고 단어로 분리
    words = text.lower().split()
    
    # 키워드 추출
    keywords = [word for word in words if word in common_keywords]
    
    # 적어도 하나의 키워드 보장
    if not keywords and len(words) > 0:
        # 가장 의미 있을 것 같은 명사 추출 (길이가 2 이상인 단어)
        for word in words:
            if len(word) >= 2 and word not in ["나는", "내가", "너무", "정말", "매우", "아주", "그냥"]:
                keywords.append(word)
    
    return keywords if keywords else ["상황"]

def analyze_note_content(text):
    """노트 내용을 분석하여 핵심 정보를 추출합니다."""
    analysis = {
        "main_topic": "",
        "severity": "중간",  # "낮음", "중간", "높음"
        "temporal_aspect": "현재",  # "과거", "현재", "미래"
        "involves_others": False,
        "action_needed": False,
        "achievement_related": False
    }
    
    # 주제 분석
    if any(word in text for word in ["업무", "일", "직장", "회사", "프로젝트"]):
        analysis["main_topic"] = "직장/업무"
    elif any(word in text for word in ["친구", "관계", "가족", "연인", "사람들"]):
        analysis["main_topic"] = "인간관계"
    elif any(word in text for word in ["건강", "몸", "아픔", "병원"]):
        analysis["main_topic"] = "건강"
    elif any(word in text for word in ["공부", "학교", "시험", "교육"]):
        analysis["main_topic"] = "교육/학습"
    elif any(word in text for word in ["돈", "재정", "금융", "지출"]):
        analysis["main_topic"] = "재정"
    else:
        analysis["main_topic"] = "일상"
    
    # 심각도 분석
    if any(word in text for word in ["너무", "매우", "극도로", "정말", "심각", "힘들", "최악"]):
        analysis["severity"] = "높음"
    elif any(word in text for word in ["조금", "약간", "살짝", "그저"]):
        analysis["severity"] = "낮음"
    
    # 시간적 측면
    if any(word in text for word in ["전에", "예전", "지난", "과거", "했었"]):
        analysis["temporal_aspect"] = "과거"
    elif any(word in text for word in ["앞으로", "미래", "계획", "예정", "할 것"]):
        analysis["temporal_aspect"] = "미래"
    
    # 다른 사람 포함 여부
    if any(word in text for word in ["친구", "동료", "상사", "팀장", "가족", "부모님", "형제", "자매", "남편", "아내"]):
        analysis["involves_others"] = True
    
    # 조치 필요 여부
    if any(word in text for word in ["해결", "방법", "어떻게", "조언", "도움"]):
        analysis["action_needed"] = True
    
    # 성취 관련 여부
    if any(word in text for word in ["성공", "달성", "완료", "성취", "이루", "해냈"]):
        analysis["achievement_related"] = True
    
    return analysis

def generate_custom_response(note, emotion, mode, keywords, analysis):
    """노트 분석 결과에 기반한 맞춤형 상담 응답을 생성합니다."""
    
    # 응답 구성 요소
    opening = ""  # 시작 문구
    feeling_part = ""  # 감정 인식 문구
    advice_part = ""  # 조언/해결책 문구
    
    # 1. 시작 문구 생성
    if mode == ResponseMode.EMOTIONAL:
        if emotion == EmotionType.SAD:
            opening_options = [
                f"{'슬픔' if '슬픔' in note else '우울함'}을 느끼고 계시는군요. ",
                f"지금 {'많이 힘드신' if analysis['severity'] == '높음' else '마음이 무거우신'} 것 같네요. ",
                f"그런 감정이 드는 건 정말 이해해요. "
            ]
        else:  # HAPPY
            opening_options = [
                f"정말 기쁜 {'일' if not analysis['achievement_related'] else '성취'}이네요! ",
                f"그런 좋은 소식이 있으셨군요! ",
                f"기분 좋은 {'경험' if not analysis['achievement_related'] else '성과'}이네요! "
            ]
    else:  # RATIONAL
        if emotion == EmotionType.SAD:
            opening_options = [
                f"{'슬픔' if '슬픔' in note else '우울함'}을 느끼고 계시는군요. ",
                f"지금 {'많이 힘드신' if analysis['severity'] == '높음' else '어려운 상황에 계신'} 것 같습니다. ",
                f"그런 감정이 드는 상황이군요. "
            ]
        else:  # HAPPY
            opening_options = [
                f"좋은 {'일' if not analysis['achievement_related'] else '성과'}가 있으셨네요. ",
                f"기쁜 {'소식' if not analysis['achievement_related'] else '성취'}이군요. ",
                f"긍정적인 {'경험' if not analysis['achievement_related'] else '결과'}을 얻으셨네요. "
            ]
    
    opening = random.choice(opening_options)
    
    # 2. 감정 인식 문구 생성
    if mode == ResponseMode.EMOTIONAL:
        if emotion == EmotionType.SAD:
            feeling_options = [
                "요즘 정말 힘드시겠네요. ",
                "그런 상황이라면 정말 마음이 무거울 수밖에 없죠. ",
                "많이 지치셨겠어요. "
            ]
        else:  # HAPPY
            feeling_options = [
                "그런 순간은 정말 소중하죠! ",
                "정말 기쁜 소식이네요. 축하해요! ",
                "그런 경험은 정말 기분 좋게 만들죠. "
            ]
    else:  # RATIONAL
        if emotion == EmotionType.SAD:
            feeling_options = [
                "그런 상황에서 힘드셨군요. 이제 어떻게 해결해 나갈지 생각해봅시다. ",
                "그런 어려움이 있으셨군요. 해결책을 같이 찾아봐요. ",
                "그런 상황이 생기면 누구나 힘들어요. 차분히 해결 방법을 찾아보죠. "
            ]
        else:  # HAPPY
            feeling_options = [
                "좋은 성과를 이루셨네요. 이 경험을 더 발전시켜 보죠. ",
                "정말 잘 하셨어요. 앞으로도 이런 성과를 이어가려면 어떻게 해야 할까요? ",
                "멋진 결과네요. 이런 성공을 계속 이어가려면 무엇이 필요할까요? "
            ]
    
    feeling_part = random.choice(feeling_options)
    
    # 3. 조언/해결책 문구 생성
    if mode == ResponseMode.EMOTIONAL:
        if emotion == EmotionType.SAD:
            if analysis["main_topic"] == "직장/업무":
                advice_options = [
                    "오늘은 작은 일부터 시작해보는 것도 좋겠어요. 잠시 휴식을 취하고 좋아하는 음료를 마시며 5분만 명상을 해보세요. 점심시간에는 잠깐 밖에 나가 햇빛을 쬐는 것도 도움이 될 거예요.",
                    "지금 업무에서 잠시 벗어나 5분 동안 심호흡을 해보세요. 오늘 퇴근 후에는 좋아하는 취미활동이나 따뜻한 목욕으로 자신을 위로해주는 건 어떨까요?",
                    "업무 스트레스가 클 때는 짧은 휴식이 중요해요. 10분만 자리에서 일어나 가벼운 스트레칭을 하고, 오늘 저녁엔 일찍 잠자리에 들어 충분한 휴식을 취해보세요."
                ]
            elif analysis["main_topic"] == "인간관계":
                advice_options = [
                    "인간관계에서 오는 스트레스는 정말 힘들어요. 오늘은 자신을 위한 시간을 가져보세요. 좋아하는 음악을 듣거나, 간단한 산책을 하며 마음을 정리해보는 건 어떨까요?",
                    "관계의 어려움이 있을 때는 스스로에게 위안을 주는 것이 중요해요. 오늘 저녁엔 좋아하는 책을 읽거나 영화를 보며 마음을 편안하게 해보세요.",
                    "때로는 잠시 거리를 두는 것도 필요해요. 혼자만의 시간을 가지고 좋아하는 활동에 집중해보세요. 따뜻한 차 한 잔과 함께 마음을 편안하게 해줄 음악을 들어보는 건 어떨까요?"
                ]
            else:
                advice_options = [
                    "오늘은 작은 일부터 시작해보는 것도 좋겠어요. 예를 들면, 짧은 산책을 하거나, 좋아하는 음악을 들어보는 건 어떨까요?",
                    "지금은 자신을 돌보는 시간이 필요할 것 같아요. 따뜻한 차 한 잔을 마시며 몇 분간 깊은 호흡을 해보세요. 저녁에는 일찍 잠자리에 들어 충분한 휴식을 취하는 것도 도움이 될 거예요.",
                    "작은 행복을 찾아보세요. 창밖을 5분만 바라보거나, 좋아하는 노래를 크게 틀어보는 것만으로도 기분이 나아질 수 있어요. 오늘 하루, 자신에게 작은 선물을 해보는 건 어떨까요?"
                ]
        else:  # HAPPY
            advice_options = [
                "이런 기쁜 마음을 일기에 기록해두거나 가까운 사람과 나누어보세요. 그리고 오늘 저녁, 작은 축하 선물로 자신이 좋아하는 디저트를 즐겨보는 건 어떨까요?",
                "이 순간의 기쁨을 더 오래 간직하기 위해 사진이나 짧은 메모로 남겨두세요. 그리고 이 기쁨을 특별한 방법으로 기념해보는 것도 좋을 것 같아요.",
                "이런 기쁜 순간을 충분히 즐기세요! 오늘 저녁에는 특별한 음식을 주문하거나, 좋아하는 활동을 하며 이 기쁨을 더 오래 즐겨보는 건 어떨까요?"
            ]
    else:  # RATIONAL
        if emotion == EmotionType.SAD:
            if analysis["main_topic"] == "직장/업무":
                advice_options = [
                    "업무 부담을 줄이기 위해 오늘 할 일 중 가장 중요한 3가지만 선택하고 집중해보세요. 각 업무 사이에 5분씩 휴식 시간을 넣고, 점심 시간에는 잠시라도 밖에 나가 걷는 시간을 가져보세요.",
                    "효율적인 업무 관리를 위해 오늘 해야 할 일을 중요도에 따라 분류해보세요. 그리고 25분 일하고 5분 휴식하는 '포모도로 기법'을 활용해보는 것도 좋습니다. 퇴근 후에는 업무 연락을 확인하지 않는 '디지털 디톡스' 시간을 가져보세요.",
                    "업무 스트레스 관리를 위해 오늘 하루만큼은 중요하지 않은 업무는 미루고, 꼭 필요한 것에만 집중해보세요. 점심 시간에 10분 명상이나 심호흡을 하고, 퇴근 후에는 가벼운 운동으로 몸의 긴장을 풀어주는 것이 효과적입니다."
                ]
            elif analysis["main_topic"] == "인간관계":
                advice_options = [
                    "갈등 상황을 해결하기 위해 오늘 짧은 메시지를 보내 대화할 의향이 있음을 전달해보세요. 대화 시에는 '나는 ~할 때 ~하게 느꼈어'라는 방식으로 감정을 표현하고, 상대방의 관점도 들어보는 것이 중요합니다.",
                    "오해를 풀기 위해 직접 만나기 어렵다면, 짧은 메시지로 먼저 연락해보세요. 상대방의 입장을 이해하려고 노력하고, 대화할 때는 구체적인 상황에 초점을 맞추는 것이 도움이 됩니다.",
                    "관계 회복을 위해 먼저 마음을 가라앉히고, 대화할 준비가 되었다면 중립적인 장소에서 만나보세요. 대화 시 비난하는 표현보다는 자신의 감정을 솔직하게 표현하는 것이 효과적입니다."
                ]
            else:
                advice_options = [
                    "상황 개선을 위해 오늘은 작은 목표 하나를 세우고 실천해보세요. 10분 명상이나 짧은 산책으로 마음을 정리하고, 취침 전에는 감사한 일 3가지를 적어보는 습관이 도움이 됩니다.",
                    "마음의 부담을 줄이기 위해 지금 당장 할 수 있는 한 가지 작은 행동부터 시작해보세요. 깊은 호흡으로 마음을 진정시키고, 하루 일과 중 짧은 휴식 시간을 반드시 가지는 것도 중요합니다.",
                    "불안한 마음을 가라앉히기 위해 오늘 10분간 마음챙김 명상을 해보세요. 긴장된 근육을 풀어주는 간단한 스트레칭과 충분한 수분 섭취도 도움이 됩니다. 저녁에는 편안한 음악과 함께 일찍 잠자리에 드는 것이 좋습니다."
                ]
        else:  # HAPPY
            if analysis["achievement_related"]:
                advice_options = [
                    "이번 성공 경험을 더 값지게 만들기 위해 성공 요소를 간단히 메모해두세요. 그리고 오늘 저녁, 작은 보상으로 자신에게 특별한 시간을 선물해보는 건 어떨까요?",
                    "이 성취감을 더 오래 간직하기 위해 오늘 있었던 일을 기록해두고, 다음 목표도 간략히 적어보세요. 그리고 이 성공을 기념하는 작은 의식을 만들어보는 것도 좋습니다.",
                    "이번 성공 경험에서 배운 점을 간단히 정리해보세요. 그리고 오늘은 자신에게 작은 보상을 주며 이 기쁨을 충분히 누려보는 건 어떨까요?"
                ]
            else:
                advice_options = [
                    "이 긍정적인 경험을 더 값지게 만들기 위해 오늘 느낀 기쁨을 일기에 적어보세요. 그리고 이 기분을 누군가와 나누어 더 풍성하게 만들어보는 건 어떨까요?",
                    "이런 좋은 경험이 주는 긍정적 에너지를 활용해 평소 미루던 일에 도전해보세요. 그리고 오늘 하루, 이 기쁨을 기념하는 작은 축하의 시간을 가져보는 것도 좋습니다.",
                    "이 기쁜 마음을 간직하기 위해 그 순간을 사진이나 글로 남겨보세요. 그리고 자신에게 작은 선물을 하며 이 긍정적인 감정을 더 오래 유지해보는 건 어떨까요?"
                ]
    
    advice_part = random.choice(advice_options)
    
    # 최종 응답 조합 - 더 간결하고 자연스럽게
    response = opening + feeling_part + advice_part
    
    return response

def generate_custom_summary(note, emotion, mode, response, analysis):
    """상담 응답에 대한 맞춤형 한 줄 요약을 생성합니다."""
    
    # 핵심 해결책 추출 (응답에서 가장 중요한 제안 부분 추출)
    key_solution = ""
    
    # 상황 기반 핵심 해결책 구성
    if analysis["main_topic"] == "직장/업무":
        if emotion == EmotionType.SAD:
            solutions = [
                "우선순위를 정하고 짧은 휴식을 챙기세요.",
                "중요한 업무 3가지만 선택하고 집중하세요.",
                "업무 사이에 짧은 휴식을 꼭 가지세요.",
                "퇴근 후에는 업무 연락을 확인하지 않는 경계를 만드세요."
            ]
        else:  # HAPPY
            solutions = [
                "성공 요소를 기록하고 다음 목표를 계획하세요.",
                "이 성취의 방법을 다른 업무에도 적용해보세요.",
                "성공을 기념하는 작은 보상을 자신에게 주세요."
            ]
    elif analysis["main_topic"] == "인간관계":
        if emotion == EmotionType.SAD:
            solutions = [
                "나-전달법으로 솔직한 대화를 시도하세요.",
                "상대방 관점을 이해하기 위해 적극적으로 경청하세요.",
                "비난 대신 감정과 상황에 초점을 맞춰 대화하세요.",
                "중립적인 장소에서 대화를 나눠보세요."
            ]
        else:  # HAPPY
            solutions = [
                "이 긍정적 경험을 소중한 사람들과 나누세요.",
                "좋은 관계의 요소를 다른 관계에도 적용해보세요.",
                "감사함을 표현하며 관계를 더 깊게 발전시키세요."
            ]
    else:  # 기타 상황
        if emotion == EmotionType.SAD:
            solutions = [
                "오늘은 작은 목표 하나를 정하고 달성해보세요.",
                "짧은 산책이나 명상으로 마음을 진정시키세요.",
                "자신에게 작은 위로와 휴식을 허락하세요.",
                "깊은 호흡으로 현재에 집중하세요."
            ]
        else:  # HAPPY
            solutions = [
                "이 긍정적인 순간을 기록하고 기념하세요.",
                "이 기쁨을 주변 사람들과 나눠보세요.",
                "이 경험에서 배운 점을 다른 상황에 활용하세요."
            ]
    
    # 랜덤하게 핵심 해결책 선택
    key_solution = random.choice(solutions)
    
    return key_solution

@v1_router.get('/health')
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

# 라우터를 애플리케이션에 포함
app.include_router(v1_router)

# 기존 엔드포인트를 하위 호환성을 위해 유지 (선택 사항)
@app.post('/chat')
def legacy_chat(req: ChatRequest):
    """하위 호환성을 위한 레거시 채팅 엔드포인트"""
    return chat(req)

@app.post('/counseling')
def legacy_counseling(req: CounselingRequest):
    """하위 호환성을 위한 레거시 상담 엔드포인트"""
    return provide_counseling(req)

@app.get('/health')
def legacy_health():
    """하위 호환성을 위한 레거시 헬스체크 엔드포인트"""
    return health_check() 