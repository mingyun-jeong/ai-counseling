import requests
import json

def test_counseling_api():
    # 테스트 케이스 목록
    test_cases = [
        {
            "name": "슬픔 감성적 대응",
            "payload": {
                "daily_note": "오늘 팀장님이 어떤 일 때문에 나한테 뭐라고 했어.",
                "emotion": "슬픔",
                "response_mode": "감성적"
            }
        },
        {
            "name": "슬픔 이성적 대응",
            "payload": {
                "daily_note": "오늘 팀장님이 어떤 일 때문에 나한테 뭐라고 했어.",
                "emotion": "슬픔",
                "response_mode": "이성적"
            }
        },
        {
            "name": "기쁨 감성적 대응",
            "payload": {
                "daily_note": "오늘 팀장님이 어떤 일 때문에 나한테 뭐라고 했어.",
                "emotion": "기쁨",
                "response_mode": "감성적"
            }
        },
        {
            "name": "기쁨 이성적 대응",
            "payload": {
                "daily_note": "오늘 팀장님이 어떤 일 때문에 나한테 뭐라고 했어.",
                "emotion": "기쁨",
                "response_mode": "이성적"
            }
        }
    ]
    
    # 각 테스트 케이스에 대해 API 호출
    for test_case in test_cases:
        print(f"\n========== 테스트: {test_case['name']} ==========")
        try:
            # API 호출
            response = requests.post(
                "http://localhost:8081/counseling",
                json=test_case["payload"],
                headers={"Content-Type": "application/json"}
            )
            
            # 응답 처리
            if response.status_code == 200:
                result = response.json()
                print(f"응답 코드: {response.status_code}")
                print("\n응답 JSON:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                print("\n✨ 상담 응답:")
                print("=" * 60)
                if 'reply' in result:
                    print(f"{result['reply']}")
                print("=" * 60)
                
                print("\n📝 요약:")
                print("=" * 60)
                if 'summary' in result:
                    print(f"{result['summary']}")
                print("=" * 60)
            else:
                print(f"오류: {response.status_code}")
                print(response.text)
            
        except Exception as e:
            print(f"연결 오류: {e}")

if __name__ == "__main__":
    print("상담 API 테스트 시작...")
    test_counseling_api() 