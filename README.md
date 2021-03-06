# Diseases Diagnostic 

๐คHuggingface Transformers 'sentence-transformers/stsb-xlm-r-multilingual' ๋ฅผ pre-train ํ์ฌ ์์ฑํ ๋ชจ๋ธ์ ์ฌ์ฉํฉ๋๋ค.
pre-train dataset์ ์์ธ๋,์์ธ ์ผ์ฑ,์์ธ ์์ฐ,๋ถ๋ฏผ ๋ณ์ ํํ์ด์ง์ ์กด์ฌํ๋ ์ง๋ณ ๋ฐฑ๊ณผ๋ฅผ ํฌ๋กค๋งํ์ต๋๋ค.

์ ๊ณต๋ ๋ชจ๋ธ์ [CLS] ํ ํฐ์ ๊ธฐ๋ฐ์ผ๋ก ๋ฌธ๋งฅ์ ์์ฑํฉ๋๋ค.

## ๐จ requirements ๐จ
1. pip install transformers
2. pip install sentence_transformers
3. pip install annoy
4. git lfs

### โ๏ธ git lfs๊ฐ ์๊ตฌ๋ฉ๋๋ค.
 12epoch_multilingual_modelํด๋์ pytorch_model.bin ํ์ผ์ ๊นํ๋ธ์ ์ฌ๋ฆด์์๋ 100MB ์ ํ์ ๋์ด์ฐ๊ธฐ ๋๋ฌธ์ lfs๋ฅผ ํตํด pushํ์์ต๋๋ค.
 ๋ฐ๋ผ์ pull์ ํ ๋์๋ lfs๋ฅผ ํด์ค์ผ ํ๊ธฐ ๋๋ฌธ์ - https://git-lfs.github.com/ - ์์ lfs๋ฅผ ์ค์นํ์ฌ ์ฃผ์๊ธธ ๋ฐ๋๋๋ค.
 
 โlfs๋ฅผ ์ฌ์ฉํ์ง ์๊ณ  clone์ ํ๊ฒ๋๋ฉด ๋ชจ๋ธ์ด ์ ์์ ์ผ๋ก ์๋์ ํ์ง์์ต๋๋ค.โ
 
 ```bash
$ git clone https://github.com/ma-an-jong/diagnostic.git
$ git lfs install
$ cd diagnostic
$ git lfs pull
```

### 1. transformers
 Tokenizer๋ฅผ ์ป๊ธฐ์ํด ์ฌ์ฉ๋ฉ๋๋ค.  ํ๊นํ์ด์ค์ 'sentence-transformers/stsb-xlm-r-multilingual' Tokenizer ๋ฅผ ์ฌ์ฉํฉ๋๋ค.

### 2. sentence_transformers
 ์ฌ์ ํ์ตํ bert๋ชจ๋ธ์ local ๊ฒฝ๋ก๋ฅผ ํตํด ๊ฐ์ ธ์ต๋๋ค. ๋ฌธ์ฅ ๋ฒกํฐ ๊ณ์ฐ์ ์ํ pooling๊ณผ ํจ๊ป SentenceTransformer๋ฅผ ์์ฑํฉ๋๋ค.

### 3. annoy
 ์ฌ์ฉ์๊ฐ ์ง๋ณ ์ ๋ณด๋ฅผ ์๋ ฅํ๋ฉด ๋ฌธ์ฅ ๋ฒกํฐ ์ ์ฌ๋ ๊ณ์ฐ์ ์ํ annoy ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ํ์๋ก ํฉ๋๋ค.

### How to Use

```python
>>> from diagnostic.disease_similarity import Model
>>> model_path = 'model_directory_path'
>>> st = Model(model_path)
>>> sentence = "๋ฐ๋ชฉ์ด ๋ถ์์ด์"
>>> indices = st.get_indices(sentence)
```
indices๋ DB์ ์กด์ฌํ๋ id๊ฐ์ ๋ฆฌํดํฉ๋๋ค. ์ง๋ณ์ ๋ํ ์กฐํ๋ฅผ ์ํ ๋๋ id๊ฐ์ key๋ก ์ฌ์ฉํ์ฌ ์กฐํํฉ๋๋ค.

score ์ธก์ ์ Modelํด๋์ค ๋ด๋ถ์ cos_sim ๋ฉ์๋๋ฅผ ์ด์ฉํฉ๋๋ค.

```python
>>> from diagnostic.disease_similarity import Model
>>> A = st.model.encode(query)
>>> B = st.model.encode(target.์ฆ์)
>>> score = st.cos_sim(A,B)
```

## Retrospection
 1. pre-train dataset์ด ๋ถ์กฑํ๋ค.
 2. ์ํ ์ ๋ฌธ ๋จ์ด ๋๋ ์์ด๋ค์ tokenizer์ ์ถ๊ฐ ํด์ผํ๋ค.
 3. ๋จ์ ๋ฒกํฐ ์ ์ฌ๋ ๋น๊ต๊ฐ ์๋ QA๋ฅผ ํตํ ํ์ต์ ํ๋ค, ์ฆ์์ ์๋ ฅํ๋ฉด ๋ ์ข์ ๋ชจ๋ธ์ด ๋์ฌ๊ฒ๊ฐ๋ค.
 4. ๊ฒ์ฆ์ ํ์ง ์์๋ค. ํ๊ฐ๋ฅผ ์ํ ๋ฐฉ๋ฒ์ด ์ ๋ ์ค๋ฅด์ง ์์๋๊ฑฐ ๊ฐ๋ค.

## Feedback
 - How to reach me : alswhd1113@gmail.com

## Reference
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Huggingface Models](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [GIT LFS](https://newsight.tistory.com/330)
