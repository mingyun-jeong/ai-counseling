import json
from datasets import load_dataset

def preprocess(input_path, output_path):
    # JSONL 형식 데이터 로드 및 전처리
    dataset = load_dataset('json', data_files=input_path)
    def format_example(ex):
        text = ex['text']
        # 감정·조언 태그 삽입
        prompt = f"<|EMPATHY|> {text} <|ADVICE|>"
        return {'prompt': prompt, 'response': ex['response']}
    processed = dataset['train'].map(format_example)
    processed.to_json(output_path, orient='records', lines=True)

if __name__ == '__main__':
    import sys
    preprocess(sys.argv[1], sys.argv[2]) 