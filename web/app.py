import os
import math
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
import pandas as pd
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dill

# --- BƯỚC 1: KHỞI TẠO ỨNG DỤNG VÀ LOAD MODEL ---
print("Khởi động server, đang tải các model...")

app = Flask(__name__)

# --- SỬA LỖI TẢI .env TẠI ĐÂY ---

# Lấy đường dẫn tuyệt đối của thư mục chứa file app.py (tức là .../web)
app_dir = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn đến thư mục cha (.../Moviesproject)
parent_dir = os.path.dirname(app_dir)

# Tạo đường dẫn đầy đủ đến file .env
dotenv_path = os.path.join(parent_dir, '.env')

print(f"Đang tải file .env từ: {dotenv_path}")

# Tải biến môi trường từ đường dẫn cụ thể đó
load_dotenv(dotenv_path=dotenv_path)

# --- KẾT THÚC SỬA LỖI ---


# Tải biến môi trường (BÂY GIỜ SẼ HOẠT ĐỘNG)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(f"Lỗi: Không tìm thấy PINECONE_API_KEY tại đường dẫn: {dotenv_path}")
else:
    print("✔️ Đã tải PINECONE_API_KEY thành công.")
    
# ... (Phần còn lại của code giữ nguyên) ...

# Kết nối tới Pinecone
print("Đang kết nối tới Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "movies" 
index = pc.Index(index_name)

# ... (Phần còn lại của code giữ nguyên) ...

# Tải model embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
print(f"Đang tải model embedding: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Tải model dịch thuật
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
print(f"Đang tải model dịch thuật: {TRANSLATION_MODEL}...")
translator = pipeline("translation", model=TRANSLATION_MODEL)

print("✅ Server sẵn sàng! Tất cả model đã được tải.")
print("="*50)


# --- BƯỚC 2: TẠO API ENDPOINT (/search) ---
@app.route('/search', methods=['GET'])
def search_movies():
    """
    API endpoint để tìm kiếm phim.
    Nhận query từ URL: /search?q=...
    """
    
    # Lấy câu truy vấn từ parameter 'q' trên URL
    vietnamese_query = request.args.get('q')
    
    # Kiểm tra nếu không có query
    if not vietnamese_query:
        return jsonify({"error": "Vui lòng cung cấp query (ví dụ: /search?q=phim gì đó)"}), 400

    print(f"Nhận request: '{vietnamese_query}'")

    try:
        # --- BƯỚC 1: DỊCH SANG TIẾNG ANH ---
        translated_result = translator(vietnamese_query)
        english_query = translated_result[0]['translation_text']
        print(f"   -> Đã dịch sang (EN): '{english_query}'")
        
        # --- BƯỚC 2: TẠO VECTOR TỪ CÂU TIẾNG ANH ---
        query_embedding = embedding_model.encode(english_query).tolist()
        
        # --- BƯỚC 3: TRUY VẤN PINECONE ---
        results = index.query(
            vector=query_embedding,
            top_k=5, # Bạn có thể đổi top_k ở đây
            include_metadata=True
        )
        
        # --- BƯỚC 4: CHUYỂN KẾT QUẢ THÀNH JSON ---
        movie_results = []
        if results['matches']:
            for match in results['matches']:
                metadata = match.get('metadata', {})
                movie_results.append({
                    "id": match.get('id', 'N/A'), # Thêm ID nếu bạn muốn
                    "title": metadata.get('title', 'N/A'),
                    "genres": metadata.get('genres', 'Không rõ'),
                    "overview": metadata.get('overview', 'Không có mô tả'),
                    "score": match.get('score', 0.0)
                })
        
        # Trả về kết quả dưới dạng JSON
        return jsonify({
            "query_vietnamese": vietnamese_query,
            "query_english": english_query,
            "results": movie_results
        })

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": "Đã xảy ra lỗi trên server"}), 500

try:
    with open('movie_recommender_2.pkl', 'rb') as file:
        movie_recommender = dill.load(file)
    print(">>> Mô hình recommender đã được tải thành công!")
except FileNotFoundError:
    print("!!! LỖI: Không tìm thấy tệp 'movie_recommender_2.pkl'.")
    print("!!! Hãy đảm bảo tệp mô hình nằm cùng thư mục với tệp API này.")
    movie_recommender = None
except Exception as e:
    # Bất kỳ lỗi nào khác (như EOFError bạn gặp)
    print(f"!!! LỖI khi tải mô hình: {e}")
    print("!!! Tệp .pkl có thể bị hỏng. Vui lòng tạo lại tệp.")
    movie_recommender = None


# ===== ĐỊNH NGHĨA ENDPOINT API =====
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    """
    Điểm cuối API để nhận đề xuất phim.
    Chấp nhận một JSON body với key 'title'.
    Ví dụ: {"title": "The Dark Knight"}
    """
    
    # 1. Kiểm tra xem mô hình đã được tải thành công chưa
    if movie_recommender is None:
        return jsonify({"error": "Mô hình recommender chưa sẵn sàng hoặc tải thất bại."}), 500

    # 2. Lấy dữ liệu JSON từ yêu cầu
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({"error": "Dữ liệu không hợp lệ. Cần một JSON body với key 'title'."}), 400

    try:
        # 3. Lấy tên phim từ JSON
        movie_title = data['title']
        
        # 4. Gọi hàm recommend() từ đối tượng mô hình của bạn
        recommendations = movie_recommender.recommend(movie_title)
        
        # 5. Trả về kết quả dưới dạng JSON
        # Giả sử hàm recommend() trả về một danh sách (list) tên phim
        return jsonify({"recommendations": recommendations.tolist()})

    except Exception as e:
        # 6. Bắt các lỗi khác (ví dụ: phim không tìm thấy)
        return jsonify({"error": f"Một lỗi đã xảy ra: {str(e)}"}), 500

# --- BƯỚC 3: CHẠY SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)