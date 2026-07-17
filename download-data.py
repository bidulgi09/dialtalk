import os
import zipfile

# 1. 환경 변수 설정 (본인의 정보로 변경하세요)
os.environ['KAGGLE_USERNAME'] = "cyatsbara"
os.environ['KAGGLE_KEY'] = os.getenv("KAGGLE_KEY")

import kaggle
def download_only_json():
    dataset_name = "chaeheonhan/kobartdata"
    download_path = "data"
    
    # 다운로드될 zip 파일의 기본 이름은 데이터셋 이름의 마지막 부분입니다.
    zip_path = os.path.join(download_path, "kobartdata.zip")
    
    os.makedirs(download_path, exist_ok=True)
    
    # 1단계: 압축 해제 없이(.zip 형태 그대로) 다운로드만 진행 (unzip=False)
    print("데이터셋 압축 파일을 다운로드합니다...")
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=False)
    
    # 2단계: 압축 파일 안에서 .json 파일만 골라서 압축 풀기
    print("다운로드 완료! JSON 파일만 골라서 추출을 시작합니다...")
    json_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # 파일 이름이 .json으로 끝나는 경우만 추출
            if file_info.filename.endswith('.json'):
                zip_ref.extract(file_info, download_path)
                json_count += 1
                
    print(f"총 {json_count}개의 JSON 파일 추출이 완료되었습니다!")
    
    # 3단계: 텍스트 파일과 함께 남아있는 원본 .zip 파일 삭제 (용량 정리)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("원본 zip 파일 삭제 완료. 깔끔하게 정리되었습니다.")

if __name__ == "__main__":
    download_only_json()