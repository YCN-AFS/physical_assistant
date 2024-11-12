# app.py
from flask import Flask, request, render_template, jsonify
import openai
from typing import List, Optional
import PyPDF2
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

app = Flask(__name__)

# Cấu trúc dữ liệu cho tài liệu
DOCUMENTS = {
    "6": {
        "Lý thuyết": [
            "Đo lường",
            "Chuyển động",
            "Lực và áp suất",
            "Năng lượng"
        ],
        "pdf_path": "documents/lop6.pdf",
        "videos": [
            {
                "title": "Vật lý 6 - Chương 1: Đo lường",
                "url": "https://youtu.be/9k559nGQL9k"
            },
            {
                "title": "Vật lý 6 - Chương 2: Chuyển động",
                "url": "https://youtu.be/tzBK-GsebMg"
            },
            {
                "title": "Vật lý 6 - Chương 3: Lực và áp suất",
                "url": "https://youtu.be/4VHmk_hTioA"
            }
        ]
    },
    "7": {
        "Lý thuyết": [
            "Nhiệt học",
            "Ánh sáng",
            "Điện học cơ bản"
        ],
        "pdf_path": "documents/lop7.pdf",
        "videos": [
            {
                "title": "Vật lý 7 - Chương 1: Nhiệt học",
                "url": "https://youtu.be/kEiyKBYWSTs"
            },
            {
                "title": "Vật lý 7 - Chương 2: Ánh sáng",
                "url": "https://youtu.be/KENE4Bt8H6Y"
            },
            {
                "title": "Vật lý 7 - Chương 3: Điện học cơ bản",
                "url": "https://youtu.be/VQRIVeBA1qQ"
            }
        ]
    },
    "8": {
        "Lý thuyết": [
            "Cơ học",
            "Nhiệt học"
        ],
        "pdf_path": "documents/lop8.pdf",
        "videos": [
            {
                "title": "Vật lý 8 - Chương 1: Cơ học",
                "url": "https://youtu.be/uJ1f7J1xuaY"
            },
            {
                "title": "Vật lý 8 - Chương 2: Nhiệt học phần 1",
                "url": "https://youtu.be/SLs5YZy9XwY"
            },
            {
                "title": "Vật lý 8 - Chương 2: Nhiệt học phần 2",
                "url": "https://youtu.be/IUxH9sINqRc"
            }
        ]
    },
    "9": {
        "Lý thuyết": [
            "Chuyển động cơ học",
            "Điện từ học",
            "Âm học",
            "Nguyên tử và hạt nhân"
        ],
        "pdf_path": "documents/lop9.pdf",
        "videos": [
            {
                "title": "Vật lý 9 - Chương 1: Chuyển động cơ học",
                "url": "https://youtu.be/kZ_tWLCq3Ck"
            },
            {
                "title": "Vật lý 9 - Chương 2: Điện từ học",
                "url": "https://youtu.be/5tgGe8j0DOg"
            },
            {
                "title": "Vật lý 9 - Chương 3: Âm học và hạt nhân",
                "url": "https://youtu.be/UwGmNAm4NlU"
            }
        ]
    }
}

class PDFPhysicsAssistant:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = {}  # Lưu trữ nội dung theo lớp
        self.embeddings = {}  # Lưu trữ embeddings theo lớp
        self.current_grade = None
        self.context = """
        Bạn là trợ lý vật lý cho học sinh. Hãy giải thích các khái niệm một cách đơn giản, 
        dễ hiểu và phù hợp với trình độ của học sinh. Chỉ trả lời dựa trên tài liệu được cung cấp.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.initialize_documents()

    def initialize_documents(self):
        """Khởi tạo và load sẵn tất cả tài liệu"""
        for grade, info in DOCUMENTS.items():
            try:
                self.load_pdf(info['pdf_path'], grade)
                self.logger.info(f"Loaded documents for grade {grade}")
            except Exception as e:
                self.logger.error(f"Error loading grade {grade}: {str(e)}")

    def load_pdf(self, pdf_path: str, grade: str, chunk_size: int = 1000):
        """Load PDF cho một lớp cụ thể"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text()

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

                return True
        except Exception as e:
            self.logger.error(f"Error loading PDF for grade {grade}: {str(e)}")
            return False

    def set_current_grade(self, grade: str):
        """Đặt lớp hiện tại để trả lời câu hỏi"""
        if grade in self.documents:
            self.current_grade = grade
            return True
        return False

    def get_relevant_docs(self, question: str, top_k: int = 3) -> List[str]:
        """Tìm các đoạn văn bản liên quan từ tài liệu của lớp hiện tại"""
        if not self.current_grade:
            return []
            
        question_embedding = self.model.encode(question)
        current_embeddings = self.embeddings[self.current_grade]
        scores = [np.dot(question_embedding, doc_emb) for doc_emb in current_embeddings]
        top_indices = np.argsort(scores)[-top_k:]
        return [self.documents[self.current_grade][i] for i in top_indices]

    def get_answer(self, question: str) -> Optional[str]:
        """Trả lời câu hỏi dựa trên tài liệu của lớp hiện tại"""
        try:
            if not self.current_grade:
                return "Vui lòng chọn lớp trước khi đặt câu hỏi."
                
            relevant_docs = self.get_relevant_docs(question)
            
            # Create a client instance
            client = openai.OpenAI(api_key=self.api_key)
            
            messages = [
                {"role": "system", "content": self.context},
                {"role": "system", "content": f"Đang trả lời cho học sinh lớp {self.current_grade}"},
                {"role": "system", "content": "Tài liệu tham khảo:\n" + "\n".join(relevant_docs)},
                {"role": "user", "content": question}
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3
            )
            
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error getting answer: {str(e)}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."

# Khởi tạo assistant
assistant = PDFPhysicsAssistant(os.getenv("OPENAI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html', documents=DOCUMENTS)

@app.route('/set_grade', methods=['POST'])
def set_grade():
    data = request.get_json()
    grade = data.get('grade')
    if assistant.set_current_grade(grade):
        return jsonify({'message': f'Đã chọn lớp {grade}'})
    return jsonify({'error': 'Lớp không hợp lệ'}), 400  

@app.route('/get_topics', methods=['POST'])
def get_topics():
    data = request.get_json()
    grade = data.get('grade')
    if grade in DOCUMENTS:
        return jsonify(DOCUMENTS[grade]['Lý thuyết'])
    return jsonify({'error': 'Lớp không hợp lệ'}), 400

@app.route('/get_videos', methods=['POST'])
def get_videos():
    data = request.get_json()
    grade = data.get('grade')
    if grade in DOCUMENTS:
        return jsonify(DOCUMENTS[grade]['videos'])
    return jsonify({'error': 'Lớp không hợp lệ'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'Chưa nhập câu hỏi'}), 400
    answer = assistant.get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)