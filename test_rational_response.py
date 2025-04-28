import requests
import json

# 기본 URL 설정
BASE_URL = "http://localhost:8081"

def test_rational_counseling():
    print("\n===== 이성적 모드 상담 API 테스트 =====")
    
    # 다양한 상황에 대한 테스트 케이스
    test_cases = [
        {
            "name": "업무 스트레스",
            "payload": {
                "daily_note": "오늘 업무가 너무 많아서 스트레스를 많이 받았어요.",
                "emotion": "슬픔",
                "response_mode": "이성적"
            }
        },
        {
            "name": "인간관계 갈등",
            "payload": {
                "daily_note": "친구와 오해가 생겨서 서로 말을 안 하고 있어요.",
                "emotion": "슬픔",
                "response_mode": "이성적"
            }
        },
        {
            "name": "성취감",
            "payload": {
                "daily_note": "오늘 중요한 프로젝트를 성공적으로 마무리했어요.",
                "emotion": "기쁨",
                "response_mode": "이성적"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n----- 테스트 케이스: {test_case['name']} -----")
        
        # API 호출
        url = f"{BASE_URL}/api/v1/counseling"
        print(f"요청 URL: {url}")
        print(f"요청 데이터: {json.dumps(test_case['payload'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(url, json=test_case['payload'])
            
            # 응답 출력
            print(f"\n상태 코드: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                
                if "reply" in result:
                    print("\n상담 응답:")
                    print("=" * 80)
                    print(result["reply"])
                    print("=" * 80)
                
                if "summary" in result:
                    print("\n요약:")
                    print("=" * 80)
                    print(result["summary"])
                    print("=" * 80)
                    
                # 이성적 모드 특징 분석
                analyze_rational_response(result)
            else:
                print(f"오류: {response.text}")
        except Exception as e:
            print(f"예외 발생: {e}")

def analyze_rational_response(result):
    """이성적 모드 응답의 특징을 분석합니다."""
    if "reply" not in result:
        return
    
    reply = result["reply"]
    
    # 특징 분석
    features = {
        "객관적 분석": any(phrase in reply for phrase in ["분석", "객관적", "생각해보", "파악"]),
        "감정 인정": any(phrase in reply for phrase in ["감정", "느끼는 것은 자연스", "이해", "공감"]),
        "성장 관점": any(phrase in reply for phrase in ["성장", "기회", "발전", "배움", "개선"]),
        "실용적 조언": any(phrase in reply for phrase in ["방법", "전략", "시도", "접근", "해결"]),
        "균형 잡힌 시각": any(phrase in reply for phrase in ["한편", "하지만", "그러나", "균형", "또한"])
    }
    
    print("\n이성적 모드 특징 분석:")
    print("-" * 50)
    for feature, present in features.items():
        status = "✓ 포함" if present else "✗ 불포함"
        print(f"{feature}: {status}")

if __name__ == "__main__":
    print("이성적 모드 상담 API 테스트 시작...\n")
    test_rational_counseling()
    print("\n테스트 완료.") 