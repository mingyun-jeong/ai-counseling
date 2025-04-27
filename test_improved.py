import requests
import json

# 테스트할 문장
test_message = "요즘 너무 우울하고 무기력해요. 아무것도 하고 싶지 않아요."

# 요청 페이로드
payload = {
    "text": test_message,
    "max_length": 200,  # 응답 길이 늘림
    "temperature": 0.1   # 온도 낮춤 (더 확정적인 응답)
}

print(f"요청: {test_message}")
print("응답 시도 중...")
print(f"요청 URL: http://localhost:8081/chat")
print(f"요청 JSON: {json.dumps(payload, indent=2, ensure_ascii=False)}")

try:
    # API 호출
    response = requests.post(
        "http://localhost:8081/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    # 응답 처리
    if response.status_code == 200:
        print(f"\n응답 코드: {response.status_code}")
        print(f"응답 헤더: {dict(response.headers)}")
        
        try:
            result = response.json()
            print("\n응답 JSON:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            print("\n응답 텍스트:")
            print("=" * 50)
            if 'reply' in result:
                print(f"{result['reply']}")
            else:
                print("응답에 'reply' 필드가 없습니다.")
            print("=" * 50)
        except Exception as json_err:
            print(f"JSON 파싱 오류: {json_err}")
            print("Raw 응답:")
            print(response.text)
    else:
        print(f"오류: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"연결 오류: {e}") 