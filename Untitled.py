#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# In[2]:


load_dotenv()


# In[8]:


import os


# In[9]:


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# In[37]:


df = pd.read_csv("./tmdb_5000_movies.csv", encoding="utf-8")


# In[38]:


df


# In[39]:


import json

def safe_parse(val):
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        return json.loads(val.replace("'", '"'))  # đổi ' thành " để JSON đọc được
    except Exception:
        try:
            return ast.literal_eval(val)  # fallback nếu JSON lỗi
        except Exception:
            return []

def build_desc(row):
    genres = [g.get("name", "") for g in safe_parse(row["genres"])]
    keywords = [k.get("name", "") for k in safe_parse(row["keywords"])]

    genres_str = ", ".join([g for g in genres if g])
    keywords_str = ", ".join([k for k in keywords if k])

    return f"""{row['title']} Genres: {genres_str} Keywords: {keywords_str} Overview: {row['overview']} Tagline: {row['tagline']}"""


# In[40]:


df["desc"] = df.apply(build_desc, axis=1)


# In[41]:


print(df["desc"][0])


# In[42]:


print(df.loc[df["title"]=="Sherlock Holmes: A Game of Shadows", "desc"].values[0])


# In[5]:


from pinecone import Pinecone
import os


# In[17]:


from transformers import pipeline


# In[6]:


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# In[37]:


def embed_text(text):
    return model.encode(text).tolist()


# In[38]:


df["embedding"] = df["desc"].apply(embed_text)


# In[39]:


print(df[["title", "embedding"]].head(2))
print("Chiều vector:", len(df["embedding"].iloc[0]))


# In[10]:


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies")


# In[47]:


import math

def clean_metadata(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""  # thay NaN bằng chuỗi rỗng
    return str(val)

vectors = []
for _, row in df.iterrows():
    vectors.append({
        "id": str(row["id"]),
        "values": row["embedding"],
        "metadata": {
            "title": clean_metadata(row["title"]),
            "overview": clean_metadata(row["overview"]),
            "tagline": clean_metadata(row["tagline"])
        }
    })


# In[49]:


def upsert_in_batches(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        index.upsert(batch)
        print(f"✅ Upsert batch {i//batch_size + 1} ({len(batch)} vectors)")


# In[50]:


upsert_in_batches(index, vectors, batch_size=100)


# In[ ]:


print(f"✅ Upsert {len(vectors)} movies vào Pinecone thành công!")


# In[24]:


translator_vi2en = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")


# In[19]:


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi")


# In[22]:


# 3️⃣ Hàm dịch
def translate_vi_to_en(text):
    if not text:
        return ""
    result = translator_vi2en(text, max_length=512)
    return result[0]['translation_text']


# In[45]:


# 4️⃣ Ví dụ query
query_en = "Sherlock Holmes: A Game of Shadows"
# query_en = translate_vi_to_en(query_vi)
# print("Query tiếng Anh:", query_en)
# 5️⃣ Encode query
q_emb = model.encode(query_en).tolist()

# 3. Query Pinecone
results = index.query(
    vector=q_emb,
    top_k=10,                # trả về 5 phim gần nhất
    include_metadata=True   # kèm theo metadata
)

# 4. In kết quả
print("🔍 Kết quả tìm kiếm:")
for match in results.matches:
    title = match.metadata.get("title", "Unknown")
    overview = match.metadata.get("overview", "")
    genres = match.metadata.get("genres", [])  # đây là list thể loại
    genres_str = ", ".join(genres) if genres else "Không có"
    
    print(f"- {title} (score: {match.score:.4f})")
    print(f"  Thể loại: {genres_str}")
    print(f"  {overview}\n")


# In[ ]:




