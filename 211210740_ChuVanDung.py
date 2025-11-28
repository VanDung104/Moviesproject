import sys
import os
from collections import Counter
import time
FILE_NAME = "dataset_de_men_phieu_luu_y_Full.txt"

MAX_PRECOMPUTED_LENGTH = 5

def load_text(filename):
    """
    Loads text from a file.
    Checks if the file exists before trying to open it.
    """
    if not os.path.exists(filename):
        print(f"Lỗi: Không tìm thấy tệp '{filename}'.")
        print("Hãy đảm bảo tệp dataset nằm cùng thư mục với tệp script này.")
        sys.exit(1)
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc tệp: {e}")
        sys.exit(1)

class CharacterPredictor:
    def __init__(self, text, max_k=MAX_PRECOMPUTED_LENGTH):
        """Khởi tạo và xây dựng mô hình tiền xử lý."""
        self.text = text
        self.text_len = len(text)
        self.max_k = max_k
        print("Đang xây dựng mô hình (preprocessing)...")
        start_time = time.time()
        # self.model sẽ có cấu trúc:
        # {
        #   0: {'': Counter({'T': 1, 'ô': 1, ...})},
        #   1: {'T': Counter({'ô': 1}), 'ô': Counter({'i': 1}), ...},
        #   2: {'Tô': Counter({'i': 1}), 'ôi': Counter({' ': 1}), ...}
        # }
        self.model = self._build_model()
        end_time = time.time()
        print(f"Xây dựng mô hình xong trong {end_time - start_time:.2f} giây.")

    def _build_model(self):
        """
        Xây dựng một mô hình N-gram (lên đến max_k) từ văn bản.
        Lưu trữ dưới dạng một dict các dict.
        """
        model = {}
        # Vòng lặp k = độ dài của bối cảnh (context)
        for k in range(self.max_k + 1):
            # sub_model chứa tất cả các context có độ dài k
            sub_model = {}
            model[k] = sub_model
            
            # Quét qua văn bản
            for i in range(self.text_len - k):
                context = self.text[i : i+k]
                next_char = self.text[i+k]
                
                # setdefault là một cách hiệu quả để thêm Counter nếu chưa có
                sub_model.setdefault(context, Counter()).update([next_char])
                
            # Thông báo tiến độ (hữu ích cho các tệp lớn)
            # print(f"Đã xử lý xong bối cảnh k={k}")
        return model

    def _predict_on_demand(self, current_input):
        """
        Đây là hàm dự đoán "chậm" từ script gốc.
        Dùng cho các chuỗi nhập dài hơn max_k (vì chúng hiếm và
        việc tìm kiếm sẽ không quá chậm).
        """
        next_char_counts = Counter()
        start_index = 0
        input_len = len(current_input)
        
        while True:
            index = self.text.find(current_input, start_index)
            if index == -1:
                break
            
            next_char_index = index + input_len
            if next_char_index < self.text_len:
                next_char = self.text[next_char_index]
                next_char_counts[next_char] += 1
            
            start_index = index + 1
            
        top_5 = next_char_counts.most_common(5)
        return [char for char, count in top_5]

    def get_predictions(self, current_input):
        """
        Lấy dự đoán. Sử dụng mô hình đã xử lý nếu có thể,
        nếu không thì dùng on-demand.
        """
        k = len(current_input)
        
        # Nếu chuỗi nhập nằm trong giới hạn đã xử lý
        if k <= self.max_k:
            # Tra cứu tức thời
            sub_model = self.model[k]
            counter = sub_model.get(current_input, Counter())
            top_5 = counter.most_common(5)
            predictions = [char for char, count in top_5]
        else:
            # Nếu chuỗi quá dài, dùng cách tìm kiếm cũ
            # print("(Dùng on-demand cho chuỗi dài)") # Debug
            predictions = self._predict_on_demand(current_input)
            
        return predictions

def main():
    """Vòng lặp tương tác chính của chương trình."""
    # Tôi đã sửa lại tên tệp ở đây để khớp với tệp bạn đã tải lên
    text_content = load_text("dataset_de_men_phieu_luu_y_ch1.txt")
    
    # Khởi tạo predictor (đây là lúc quá trình preprocessing xảy ra)
    predictor = CharacterPredictor(text_content, max_k=MAX_PRECOMPUTED_LENGTH)
    
    current_input = ""
    
    print("-" * 50)
    print("CHƯƠNG TRÌNH DỰ ĐOÁN KÝ TỰ (BẢN TỐI ƯU)")
    print("Gõ một ký tự để bắt đầu.")
    print("Nhấn [Enter] (không gõ gì) để nhập lại từ đầu.")
    print("Gõ '0' và nhấn [Enter] để quay lại (xoá 1 ký tự).") # <-- HƯỚNG DẪN MỚI
    print("Nhấn Ctrl+C để thoát chương trình.")
    print("-" * 50)

    while True:
        try:
            # Lấy dự đoán - giờ đây sẽ rất nhanh
            predictions = predictor.get_predictions(current_input)
            
            pred_display = [repr(p) for p in predictions]
            
            print(f"\nChuỗi hiện tại: '{current_input}'")
            print(f"5 gợi ý hàng đầu: {', '.join(pred_display)}")
            
            char_input = input("Nhập ký tự tiếp theo: ")
            
            if not char_input:
                # Nếu người dùng nhấn Enter (chuỗi rỗng) -> Reset
                current_input = ""
                print("\n--- Đã reset ---")
                continue
            
            # --- LOGIC MỚI ĐỂ QUAY LẠI ---
            if char_input == '0':
                if current_input: # Chỉ xoá nếu chuỗi không rỗng
                    current_input = current_input[:-1] # Xoá ký tự cuối
                    print(f"\n--- Đã xoá ký tự cuối ---")
                else:
                    print(f"\n--- Chuỗi rỗng, không thể xoá ---")
                continue # Quay lại đầu vòng lặp để lấy dự đoán mới
            # --- KẾT THÚC LOGIC MỚI ---
            
            # Nếu không phải '0' hoặc chuỗi rỗng, thêm ký tự đầu tiên
            next_char = char_input[0]
            current_input += next_char
            
        except KeyboardInterrupt:
            print("\nĐang thoát chương trình. Tạm biệt!")
            break
        except Exception as e:
            print(f"\nĐã xảy ra lỗi: {e}")
            break

if __name__ == "__main__":
    main()