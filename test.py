import os
import logging
import PyPDF2
import openai
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
import numpy as np
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cấu hình tài liệu cho từng lớp
DOCUMENTS = {
    "6": {
        "Lý thuyết": [
            "Đo lường",
            "Chuyển động",
            "Lực và áp suất",
            "Năng lượng"
        ],
        "pdf_path": "documents/lop6.pdf"
    },
    "7": {
        "Lý thuyết": [
            "Nhiệt học",
            "Ánh sáng",
            "Điện học cơ bản"
        ],
        "pdf_path": "documents/lop7.pdf"
    },
    "8": {
        "Lý thuyết": [
            "Công và công suất",
            "Điện học nâng cao",
            "Từ trường"
        ],
        "pdf_path": "documents/lop8.pdf"
    },
    "9": {
        "Lý thuyết": [
            "Chuyển động cơ học",
            "Điện từ học",
            "Âm học",
            "Nguyên tử và hạt nhân"
        ],
        "pdf_path": "documents/lop9.pdf"
    }
}


class PDFPhysicsAssistant:
    def __init__(self, api_key: str):
        """Khởi tạo assistant với OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key
        
        self.logger = logging.getLogger(__name__)
        
        # Load model sentence transformer
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Lưu trữ documents và embeddings theo lớp
        self.documents: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, List] = {}
        
        # Lớp hiện tại đang được sử dụng
        self.current_grade: Optional[str] = None
        
        # Context mặc định cho assistant
        self.context = """Bạn là một trợ lý thông minh giúp học sinh học vật lý. 
        Hãy trả lời các câu hỏi dựa trên tài liệu được cung cấp một cách chính xác và dễ hiểu.
        Nếu câu hỏi không liên quan đến nội dung trong tài liệu, hãy từ chối trả lời một cách lịch sự."""
        
        # Khởi tạo documents
        self.initialize_documents()

    def initialize_documents(self):
        """Khởi tạo và load sẵn tất cả tài liệu"""
        # Tạo thư mục documents nếu chưa tồn tại
        documents_dir = "documents"
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)
            self.logger.info(f"Đã tạo thư mục {documents_dir}")

        for grade, info in DOCUMENTS.items():
            try:
                pdf_path = info['pdf_path']
                if not os.path.exists(pdf_path):
                    self.logger.error(f"Không tìm thấy file: {pdf_path}")
                    continue
                    
                success = self.load_pdf(pdf_path, grade)
                if success:
                    self.logger.info(f"Đã load thành công tài liệu lớp {grade}")
                else:
                    self.logger.error(f"Không load được tài liệu lớp {grade}")
            except Exception as e:
                self.logger.error(f"Lỗi khởi tạo lớp {grade}: {str(e)}")

    def load_pdf(self, pdf_path: str, grade: str, chunk_size: int = 1000) -> bool:
        """Load PDF cho một lớp cụ thể"""
        try:
            if not os.path.exists(pdf_path):
                self.logger.error(f"Không tìm thấy file PDF: {pdf_path}")
                return False
                
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if len(pdf_reader.pages) == 0:
                        self.logger.error(f"PDF không có trang nào: {pdf_path}")
                        return False
                        
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    if not text.strip():
                        self.logger.error(f"Không trích xuất được text từ PDF: {pdf_path}")
                        return False

                    # Chia thành các đoạn
                    words = text.split()
                    chunks = []
                    current_chunk = []
                    
                    for word in words:
                        current_chunk.append(word)
                        if len(current_chunk) >= chunk_size:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                    
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))

                    # Lưu trữ theo lớp
                    self.documents[grade] = chunks
                    self.embeddings[grade] = [self.model.encode(chunk) for chunk in chunks]
                    
                    self.logger.info(f"Đã load thành công PDF cho lớp {grade}: {len(chunks)} chunks")
                    return True

                except PyPDF2.PdfReadError as e:
                    self.logger.error(f"Lỗi đọc PDF {pdf_path}: {str(e)}")
                    return False

        except Exception as e:
            self.logger.error(f"Lỗi load PDF cho lớp {grade}: {str(e)}")
            return False

    def set_grade(self, grade: str) -> bool:
        """Thiết lập lớp hiện tại"""
        if grade not in DOCUMENTS:
            return False
        self.current_grade = grade
        return True

    def get_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Lấy các đoạn văn bản liên quan nhất với câu hỏi"""
        if not self.current_grade:
            return []
            
        query_embedding = self.model.encode(query)
        
        # Tính similarity với tất cả chunks trong lớp hiện tại
        similarities = []
        for doc_embedding in self.embeddings[self.current_grade]:
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append(similarity)
            
        # Lấy top_k chunks có similarity cao nhất
        top_indices = np.argsort(similarities)[-top_k:]
        
        return [self.documents[self.current_grade][i] for i in top_indices]

    def get_answer(self, question: str) -> Optional[str]:
        """Trả lời câu hỏi dựa trên tài liệu của lớp hiện tại"""
        try:
            if not self.current_grade:
                return "Vui lòng chọn lớp trước khi đặt câu hỏi."
                
            if not question.strip():
                return "Vui lòng nhập câu hỏi."

            # Kiểm tra xem đã load được tài liệu chưa
            if not self.documents.get(self.current_grade):
                return "Chưa load được tài liệu cho lớp này."

            relevant_docs = self.get_relevant_docs(question)
            if not relevant_docs:
                return "Không tìm thấy thông tin liên quan trong tài liệu."

            messages = [
                {"role": "system", "content": self.context},
                {"role": "system", "content": f"Đang trả lời cho học sinh lớp {self.current_grade}"},
                {"role": "system", "content": "Tài liệu tham khảo:\n" + "\n".join(relevant_docs)},
                {"role": "user", "content": question}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content

        except openai.error.AuthenticationError:
            self.logger.error("Lỗi xác thực API key")
            return "Lỗi xác thực API key. Vui lòng kiểm tra lại cấu hình."
        
        except openai.error.RateLimitError:
            self.logger.error("Đã vượt quá giới hạn API")
            return "Hệ thống đang bận. Vui lòng thử lại sau."
        
        except openai.error.APIError as e:
            self.logger.error(f"Lỗi API OpenAI: {str(e)}")
            return "Có lỗi khi gọi API. Vui lòng thử lại sau."
        
        except Exception as e:
            self.logger.error(f"Lỗi không xác định: {str(e)}")
            return f"Xin lỗi, có lỗi xảy ra: {str(e)}"

# Khởi tạo Flask app
app = Flask(__name__)

# Khởi tạo assistant
assistant = PDFPhysicsAssistant(os.getenv("OPENAI_API_KEY"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/set_grade', methods=['POST'])
def set_grade():
    """API endpoint để thiết lập lớp"""
    grade = request.json.get('grade')
    if not grade:
        return jsonify({"error": "Thiếu tham số grade"}), 400
        
    if assistant.set_grade(grade):
        return jsonify({"message": f"Đã chọn lớp {grade}"})
    else:
        return jsonify({"error": f"Lớp không hợp lệ: {grade}"}), 400

@app.route('/ask', methods=['POST'])
def ask():
    """API endpoint để đặt câu hỏi"""
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Thiếu tham số question"}), 400
        
    answer = assistant.get_answer(question)
    return jsonify({"answer": answer})

@app.route('/status')
def status():
    """API endpoint để kiểm tra trạng thái hệ thống"""
    status = {
        "documents_loaded": list(assistant.documents.keys()),
        "current_grade": assistant.current_grade,
        "api_key_set": bool(assistant.api_key)
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)