# Diseases Diagnostic 

🤗Huggingface Transformers 'sentence-transformers/stsb-xlm-r-multilingual' 를 pre-train 하여 생성한 모델을 사용합니다.
pre-train dataset은 서울대,서울 삼성,서울 아산,부민 병원 홈페이지에 존재하는 질병 백과를 크롤링하여 수집한 약 4만줄의 문장을 12 epochs 동안 학습 시켰습니다.

## 🚨 requirements 🚨
1. pip install transformers
2. pip install sentence_transformers
3. pip install annoy

### 1. transformers
 Tokenizer를 얻기위해 사용됩니다.  허깅페이스의 'sentence-transformers/stsb-xlm-r-multilingual' Tokenizer 를 사용합니다.

### 2. sentence_transformers
 사전학습한 bert모델을 local 경로를 통해 가져옵니다. 문장 벡터 계산을 위한 pooling과 함께 SentenceTransformer를 생성합니다.

### 3. annoy
 사용자가 질병 정보를 입력하면 문장 벡터 유사도 계산을 위한 annoy 라이브러리를 필요로 합니다.

### How to Use

```python
>>> from diagnostic.disease_similarity import Model
>>> st = Model()
>>> sentence = 발목이 부었어요"
>>> indices = st.get_indices(sentence)
```
indices는 DB에 존재하는 id값을 리턴합니다. 질병에 대한 조회를 원할때는 id값을 key로 사용하여 조회합니다.


score 측정은 Model클래스 내부의 cos_sim 메소드를 이용합니다.

```python
>>> from diagnostic.disease_similarity import Model
>>> A = st.model.encode(query)
>>> B = st.model.encode(target.증상)
>>> score = st.cos_sim(A,B)
```

## Reference
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Huggingface Models](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual)
