# S&P 100 공시문서 기반 RAG 시스템

## 📋 프로젝트 개요
S&P 100 기업들의 SEC 공시문서(10-K)를 활용하여 기업 정보를 정확하게 검색하고 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 🎬 시연 연상
[Demo video link](https://youtu.be/LBBBavYPNJM)

## 🚀 주요 특징
- **자동화된 데이터 수집**: SEC EDGAR API를 통한 최신 10-K 문서 크롤링
- **지능형 문서 처리**: HTML 파싱 및 의미있는 섹션 추출
- **최적화된 검색**: 자체 개발한 가중치 기반 랭킹 알고리즘
- **Cross-Encoder Reranking**: 검색 정확도 향상을 위한 2단계 랭킹

## 🛠️ 기술 스택
- **Language**: Python
- **Embedding**: OpenAI Text Embedding API
- **Vector DB**: FAISS
- **Frontend**: Streamlit
- **Processing**: Regex

## 📁 프로젝트 구조

### 1. `crawler.py` - 데이터 수집
S&P 100 ETF(iShares) 구성 기업의 최근 3년간(2023-2025) 10-K 문서(연례보고서)를 자동 다운로드

<details>
<summary>상세 내용 보기</summary>

- **데이터 소스**: SEC EDGAR API
- **대상 문서**: 10-K (연례보고서)
- **저장 위치**: `10k_raw/` 폴더
- **특징**: 
  - 최근 S&P 100 구성종목 반영
  - API 제한 준수

</details>

### 2. `cleaner.py` - 문서 전처리
원본 HTML을 파싱하고, 투자자에게 중요한 정보가 담긴 Part 1, 2만 추출 및 정제하여, json 파일로 저장

<details>
<summary>상세 내용 보기</summary>

- **주요 기능**:
  - 정규표현식 기반 섹션 추출
  - HTML/iXBRL 태그 제거
  - 표를 텍스트로 변환
  - 길이에 기반하여, 섹션별 중요도(weight) 계산
- **출력 형식**: JSON
  ```json
  {
    "ticker": "AAPL",
    "filing_year": 2024,
    "part": 1,
    "section_item": "Item 1",
    "section_title": "Business",
    "section_weight": 0.35,
    "text": "..."
  }
  ```

</details>

### 3. `chunker.py` - 텍스트 청킹
문맥을 보존하면서 임베딩에 최적화된 크기로 문서 분할

<details>
<summary>상세 내용 보기</summary>

- **청킹 전략**:
  - 문장 단위 분할 (문맥 보존)
  - 동적 청크 크기: 길이 비중에 따라 중요도를 매겼으며, 섹션 중요도와 청크 크기가 반비례하도록 가변 설정
  - 청크 크기 범위: 256-1024 토큰
- **특징**:
  - 중요 섹션은 작은 청크로 세밀하게 분할
  - 오버랩 설정 조절 가능
  - 테이블/긴 문장 강제 분할

</details>

### 4. `embedder.py` - 벡터 임베딩
OpenAI API를 사용한 텍스트 임베딩 및 FAISS 인덱싱

<details>
<summary>상세 내용 보기</summary>

- **임베딩 모델**: OpenAI text-embedding-3-large
- **벡터 DB**: FAISS (Facebook AI Similarity Search)
- **유사도 메트릭**: Cosine Similarity
- **인덱스 저장**: 루트 폴더 밑 `faiss/` 폴더

</details>

### 5. `app.py` & `reranker.py` - 검색 및 서빙
Streamlit 기반 UI와 고도화된 검색 로직

<details>
<summary>상세 내용 보기</summary>

- **검색 프로세스**:
  1. 쿼리에서 기업명(ticker) 자동 추출
  2. Cosine Similarity 기반 1차 검색 (Top K1)
  3. 타겟 기업 청크에 가중치(α) 부여
  4. Cross-Encoder 기반 리랭킹 (Top K2)
  5. 최종 Top 10 청크로 답변 생성
- **UI 특징**:
  - 파라미터(K1, K2, M) 조절 가능
  - 섹터별 기업 분류 표시
  - 답변에서 참고 문서 소스 확인 가능

</details>

### 6. `evaluation/` - 성능 평가 및 최적화 🌟
**자체 평가 프레임워크 구축으로 검색 정확도 향상**

<details>
<summary>상세 내용 보기</summary>

#### 문제 정의
- 단순 Cosine Similarity의 한계: 의미는 유사하나 다른 기업의 정보가 상위 랭크됨
- 예시: "Nvidia의 AI 전략" 검색 시 → Broadcom의 AI 전략이 나오는 문제

#### 해결 방안
1. **가중치 시스템 도입**: 쿼리에 언급된 기업의 문서로부터 나온 청크에 가산점(α) 부여
2. **자동 평가 데이터셋 구축**:
   - `generate_query.py`: 섹터별 대표 쿼리 생성
   - `label_query.py`: LLM을 활용한 청크의 쿼리에 대한 답변 가능성 라벨링
3. **최적화**:
   - `alpha_optimization.py`: Grid Search로 최적 α값 탐색
   - **결과**: α=0.2에서 최고 Recall 달성

#### 성과
- 타겟 기업 정보 검색 정확도 향상(Recall 기준 약 15% 향상)
- 연관 기업 정보도 함께 제공하는 균형잡힌 검색 방법

</details>

## 💡 프로젝트 하이라이트

1. **청크 크기 가변적으로 설계**: 공시문서의 특성을 고려한 섹션별 가중치 시스템
2. **확장 가능한 구조**: 모듈화된 파이프라인으로 쉬운 기능 추가
3. **타겟 기업 추출**: 쿼리에서 타겟한 기업에 대해 명확한 답변 가능
4. **답변 소스 확인 가능**: 답변에서 어떤 공시문서의 어느 부분을 참조했는지 확인 가능

## 🚀 실행 순서 (자세한 argument 사용은 각 스크립트의 상단 주석 참고)

```bash
# 1. 데이터 수집
python crawler.py 

# 2. 전처리
python cleaner.py

# 3. 청킹
python chunker.py

# 4. 임베딩(OpenAI API Key 필요)
python embedder.py

# 5. 웹 애플리케이션 실행(OpenAI API Key 필요)
streamlit run app.py
```
