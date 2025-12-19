import os
import time
import json
import dill  # Thêm thư viện này
import pandas as pd # Thêm pandas
from sklearn.feature_extraction.text import TfidfVectorizer # Thêm sklearn
from sklearn.metrics.pairwise import cosine_similarity      # Thêm sklearn

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# --- IMPORTS CHO AI & LANGCHAIN ---
from pinecone import Pinecone
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 

DOTENV_PATH = os.path.join(parent_dir, '.env')
# Đảm bảo file pickle nằm đúng vị trí này
PICKLE_FILE_PATH = os.path.join(parent_dir, 'movie_recommender_2.pkl') 

load_dotenv(dotenv_path=DOTENV_PATH)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Vui lòng kiểm tra file .env, thiếu API KEY!")

# ==============================================================================
# 1. ĐỊNH NGHĨA CLASS MOVIE RECOMMENDER (Bắt buộc để load pickle)
# ==============================================================================
class MovieRecommender:
    def __init__(self, data_path):
        # Đọc dữ liệu
        self.df = pd.read_csv(data_path)
        
        # Xử lý genres và keywords thành chuỗi
        self.df['string'] = self.df.apply(self.genres_and_keywords_to_string, axis=1)
        
        # Tạo TF-IDF vectorizer và fit-transform dữ liệu
        self.tfidf = TfidfVectorizer(max_features=2000)
        self.X = self.tfidf.fit_transform(self.df['string'])
        
        # Tạo Series để dễ dàng truy xuất movie index
        self.movie_idx = pd.Series(self.df.index, index=self.df['title'])
    
    @staticmethod
    def genres_and_keywords_to_string(row):
        genres = json.loads(row['genres'])
        genres = ' '.join(''.join(j['name'].split()) for j in genres)

        keywords = json.loads(row['keywords'])
        keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
        return "%s %s" % (genres, keywords)
    
    def recommend(self, title):
        if title not in self.movie_idx:
            return None # Trả về None để API xử lý thông báo lỗi
        else:
            idx = self.movie_idx[title]
            if type(idx) == pd.Series:  # Kiểm tra xem có nhiều phim trùng tên không
                idx = idx.iloc[0]
            
            query = self.X[idx]
            scores = cosine_similarity(query, self.X)
            scores = scores.flatten()
            recommended_idx = (-scores).argsort()[1:6]
            return self.df['title'].iloc[recommended_idx]

# ==============================================================================
# 2. KHỞI TẠO FLASK & LOAD MODELS
# ==============================================================================
app = Flask(__name__)
CORS(app)

print("--- ĐANG KHỞI TẠO HỆ THỐNG ---")

# A. Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "movies"
index = pc.Index(index_name)
print(f"✅ Đã kết nối Pinecone Index: {index_name}")

# B. Translation Model
print("⏳ Đang tải model dịch thuật...")
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
translator = pipeline("translation", model=TRANSLATION_MODEL)
print("✅ Model dịch thuật sẵn sàng.")

# C. Embedding Model
print("⏳ Đang tải model Embedding...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("✅ Model Embedding sẵn sàng.")

# D. LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)

# E. TF-IDF Recommender (Phần mới thêm)
print("⏳ Đang tải model TF-IDF (Pickle)...")
movie_recommender = None
try:
    if os.path.exists(PICKLE_FILE_PATH):
        with open(PICKLE_FILE_PATH, 'rb') as file:
            movie_recommender = dill.load(file)
        print("✅ Model TF-IDF đã được tải thành công.")
    else:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy file {PICKLE_FILE_PATH}")
except Exception as e:
    print(f"❌ Lỗi khi tải file pickle: {str(e)}")
    # Nếu lỗi, movie_recommender sẽ là None

# ==============================================================================
# 3. CHAIN LANGCHAIN (Giữ nguyên)
# ==============================================================================
def translate_query(vietnamese_query):
    if not vietnamese_query: return ""
    print(f"[PROCESS] 1. Dịch: '{vietnamese_query}'")
    translated_result = translator(vietnamese_query)
    english_query = translated_result[0]['translation_text']
    print(f"   -> Kết quả: '{english_query}'")
    return english_query

def get_embedding(english_query):
    print(f"[PROCESS] 2. Tạo Vector")
    return embedding_model.encode(english_query).tolist()

def query_pinecone(query_embedding):
    print("[PROCESS] 3. Truy vấn Pinecone")
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    return results['matches']

def format_context_from_pinecone(matches):
    print("[PROCESS] 4. Format Context")
    if not matches:
        return ""
    
    context = ""
    for i, match in enumerate(matches):
        score = match['score']
        metadata = match.get('metadata', {})
        title = metadata.get('title', 'N/A')
        genres = metadata.get('genres', 'Không rõ')
        overview = metadata.get('overview', 'Không có tóm tắt.')
        
        context += f"--- Phim {i+1} ---\n"
        context += f"Tiêu đề: {title}\n"
        context += f"Điểm tương đồng: {score:.2f}\n"
        context += f"Thể loại: {genres}\n"
        context += f"Tóm tắt: {overview}\n\n"
    print(context)
    return context

translate_step = RunnableLambda(translate_query)
embed_step = RunnableLambda(get_embedding)
query_pinecone_step = RunnableLambda(query_pinecone)
augmentation_step = RunnableLambda(format_context_from_pinecone)

retrieval_chain = (translate_step | embed_step | query_pinecone_step | augmentation_step)

prompt_template = """
Bạn là một trợ lý tư vấn phim ảnh chuyên nghiệp.
Người dùng đang tìm kiếm phim với mô tả: "{question}"

Đây là danh sách các phim phù hợp nhất tìm thấy từ cơ sở dữ liệu:
{context}

**NHIỆM VỤ:**
Nếu không có phim nào trong danh sách (context rỗng), hãy xin lỗi người dùng.
Nếu có phim, hãy viết câu trả lời thân thiện bằng Tiếng Việt:
1. Xác nhận nhu cầu tìm kiếm của họ.
3. Với mỗi phim, nêu rõ: Tên phim, Thể loại, Tóm tắt ngắn và LÝ DO tại sao nó phù hợp.
Nếu phim nào có điểm tương đồng dưới 0.6 thì trả lời không.
Hãy trả lời ngắn gọn, súc tích và trình bày đẹp (dùng Markdown).
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()

rag_chain = ({"context": retrieval_chain, "question": RunnablePassthrough()} | prompt | llm | parser)

# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        raw_query = data.get('query', '')
        
        print(f"\n📩 Nhận request: {raw_query}")

        response_text = ""
        tool_used = "none"

        # --- MODE 1: SEMANTIC SEARCH (RAG) ---
        if "[MODE: SEMANTIC_SEARCH]" in raw_query:
            real_query = raw_query.replace("[MODE: SEMANTIC_SEARCH]", "").strip()
            if real_query:
                response_text = rag_chain.invoke(real_query)
            else:
                response_text = "Bạn vui lòng nhập nội dung cần tìm kiếm nhé!"
            tool_used = "semantic"

        # --- MODE 2: TF-IDF RECOMMEND (Phần mới thêm) ---
        elif "[MODE: TFIDF_RECOMMEND]" in raw_query:
            movie_title = raw_query.replace("[MODE: TFIDF_RECOMMEND]", "").strip()
            
            # Kiểm tra xem model đã load được chưa
            if movie_recommender is None:
                response_text = "Hệ thống đang gặp sự cố tải dữ liệu gợi ý (file pickle không khả dụng)."
            else:
                # Gọi hàm recommend từ object đã load
                print(f"[TF-IDF] Tìm kiếm phim tương tự: {movie_title}")
                recommendations = movie_recommender.recommend(movie_title)

                if recommendations is None:
                    response_text = f"Xin lỗi, tôi không tìm thấy bộ phim **'{movie_title}'** trong cơ sở dữ liệu để đưa ra gợi ý."
                else:
                    # Format danh sách phim trả về (Không dùng LLM)
                    list_items = "\n".join([f"🎬 **{title}**" for title in recommendations])
                    response_text = (
                        f"Dựa trên thuật toán TF-IDF, dưới đây là 5 bộ phim có nội dung/thể loại tương tự với **{movie_title}**:\n\n"
                        f"{list_items}"
                    )
            
            tool_used = "tfidf"

        else:
            response_text = "Hệ thống không nhận diện được chế độ. Vui lòng chọn chế độ tìm kiếm."

        return jsonify({
            "response": response_text,
            "tool_used": tool_used
        })

    except Exception as e:
        print(f"❌ LỖI SERVER: {str(e)}")
        return jsonify({
            "response": f"Đã xảy ra lỗi phía server: {str(e)}",
            "tool_used": "none"
        }), 500

@app.route('/')
def home():
    return send_from_directory(current_dir, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)