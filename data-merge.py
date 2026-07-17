import os
import json
import glob

def integrate_dialect_data(input_directory, output_filename):
    """
    여러 개의 사투리 대화 JSON 파일을 하나의 포맷팅된 JSON 파일로 통합합니다.
    """
    # 통합된 데이터를 저장할 리스트
    integrated_data = []

    # 입력 디렉토리 내의 모든 JSON 파일 경로를 가져옴
    # 실제 데이터가 들어있는 폴더 경로로 변경해야 합니다. (예: "./json_data/*.json")
    file_pattern = os.path.join(input_directory, "*.json")
    file_list = glob.glob(file_pattern)

    print(f"총 {len(file_list)}개의 파일을 찾았습니다. 데이터 추출을 시작합니다...")

    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # json 구조에서 'utterance' 리스트에 접근
                utterances = data.get("utterance", [])
                
                for utter in utterances:
                    # dialect_form과 standard_form 추출
                    dialect = utter.get("dialect_form", "").strip()
                    standard = utter.get("standard_form", "").strip()
                    
                    # 두 값이 모두 존재하는 경우에만 리스트에 추가
                    if dialect and standard:
                        integrated_data.append({
                            "dialect": dialect,
                            "standard": standard
                        })
                        
        except Exception as e:
            print(f"파일을 처리하는 중 오류가 발생했습니다 ({file_path}): {e}")

    # 최종 데이터를 하나의 JSON 파일로 저장
    try:
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            # ensure_ascii=False 를 설정해야 한글이 유니코드(제대로) 깨지지 않고 저장됩니다.
            json.dump(integrated_data, out_f, ensure_ascii=False, indent=4)
        print(f"\n성공적으로 변환이 완료되었습니다! 총 {len(integrated_data)}개의 문장이 '{output_filename}'에 저장되었습니다.")
    except Exception as e:
        print(f"파일을 저장하는 중 오류가 발생했습니다: {e}")

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 1. 원본 7699개의 JSON 파일들이 모여있는 폴더 경로를 지정하세요. 
    INPUT_DIR = "data" 
    
    # 2. 결과물로 만들어질 통합 JSON 파일의 이름을 지정하세요.
    OUTPUT_FILE = "data.json"
    
    integrate_dialect_data(INPUT_DIR, OUTPUT_FILE)