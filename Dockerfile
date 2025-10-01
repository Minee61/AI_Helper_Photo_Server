# 베이스 이미지
FROM python:3.10-slim

# 작업 디렉토리
WORKDIR /code

# 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 환경 변수
ENV PORT=7860
EXPOSE 7860

# 서버 실행
CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port $PORT"]
