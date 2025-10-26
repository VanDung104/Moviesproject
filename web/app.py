import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
from flask import Flask, request, jsonify

# --- KHỞI TẠO ỨNG DỤNG FLASK ---
app = Flask(__name__)

# --- CẤU HÌNH VÀ TẢI MÔ HÌNH (CHỈ MỘT LẦN KHI START) ---

# 1. Tải biến môi trường từ file .env ở thư mục gốc của dự án
# Đường dẫn từ 'web/app.py' đến thư mục cha 'Moviesproject' là '..'
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Vui lòng đặt PINECONE_API_KEY trong file .env")

# 2. Tải các mô hình và kết nối dịch vụ
print("=> Đang tải mô hình và kết nối dịch vụ. Vui lòng đợi...")
try:
    # Mô hình để tạo vector embedding
    print("   - Tải SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Mô hình dịch thuật
    print("   - Tải Translation model...")
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    translator = pipeline("translation", model=TRANSLATION_MODEL)

    # Kết nối tới Pinecone
    print("   - Kết nối tới Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "movies"
    
    # Kiểm tra xem index có tồn tại không
    if index_name not in pc.list_indexes().names():
         raise NameError(f"Index '{index_name}' không tồn tại trong Pinecone. Vui lòng tạo index trước.")
         
    index = pc.Index(index_name)
    print("=> Khởi tạo hoàn tất. Máy chủ đã sẵn sàng! ✅")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo mô hình hoặc kết nối dịch vụ. {e}")
    # Thoát nếu không thể tải mô hình
    exit()


# --- LOGIC TÌM KIẾM PHIM (Đã chỉnh sửa để trả về JSON) ---
def find_similar_movies(vietnamese_query, top_k=5):
    """
    Hàm tìm kiếm đã được sửa đổi để trả về một danh sách các dictionary.
    """
    print(f"\n🎬 Nhận yêu cầu tìm kiếm cho: '{vietnamese_query}'")

    # BƯỚC 1: DỊCH SANG TIẾNG ANH
    translated_result = translator(vietnamese_query)
    english_query = translated_result[0]['translation_text']
    print(f"   -> Đã dịch sang (EN): '{english_query}'")

    # BƯỚC 2: TẠO VECTOR TỪ CÂU TIẾNG ANH
    query_embedding = model.encode(english_query).tolist()

    # BƯỚC 3: TRUY VẤN PINECONE
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # BƯỚC 4: CHUẨN BỊ KẾT QUẢ ĐỂ TRẢ VỀ
    movie_results = []
    if not results['matches']:
        print("   -> Không tìm thấy kết quả phù hợp.")
        return []

    for match in results['matches']:
        movie_results.append({
            'title': match['metadata'].get('title', 'N/A'),
            'overview': match['metadata'].get('overview', 'No overview available.'),
            'score': match['score']
        })
    print(f"   -> Tìm thấy {len(movie_results)} kết quả.")
    return movie_results


# --- ĐỊNH NGHĨA API ENDPOINT ---
@app.route('/search', methods=['GET'])
def search():
    """
    Endpoint nhận query từ người dùng và trả về kết quả dưới dạng JSON.
    Cách dùng: GET /search?query=phim về rồng
    """
    # Lấy query từ URL parameter
    user_query = request.args.get('query')

    # Kiểm tra nếu không có query
    if not user_query:
        return jsonify({"error": "Vui lòng cung cấp 'query' parameter."}), 400

    try:
        # Gọi hàm tìm kiếm
        similar_movies = find_similar_movies(user_query)
        # Trả về kết quả dưới dạng JSON
        return jsonify({"results": similar_movies})
    except Exception as e:
        print(f"LỖI API: {e}")
        return jsonify({"error": "Đã có lỗi xảy ra trên máy chủ."}), 500

@app.route('/', methods=['GET'])
def home():
    return "<h1>Movie Search API</h1><p>Sử dụng endpoint <code>/search?query=</code> để tìm kiếm.</p>"

# --- CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    # Chạy server trên cổng 5000, có thể truy cập từ mọi IP
    app.run(host='0.0.0.0', port=5000, debug=True)

