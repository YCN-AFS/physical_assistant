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
    "physics": {
        "6": {
            "Lý thuyết": [
                "Đo lường",
                "Chuyển động",
                "Lực và áp suất",
                "Năng lượng"
            ],
            "pdf_path": "documents/physics/lop6.pdf",
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
            "pdf_path": "documents/physics/lop7.pdf",
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
            "pdf_path": "documents/physics/lop8.pdf",
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
            "pdf_path": "documents/physics/lop9.pdf",
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
    },
    "chemistry": {
        "6": {
            "Lý thuyết": [
                "Chất và sự biến đổi của chất",
                "Không khí và sự cháy",
                "Nước và dung dịch"
            ],
            "pdf_path": "documents/chemistry/lop6.pdf",
            "videos": []
        },
        # ... thêm nội dung cho các lớp khác ...
    },
    "biology": {
        "6": {
            "Lý thuyết": [
                "Tế bào và vi sinh vật",
                "Thực vật",
                "Động vật"
            ],
            "pdf_path": "documents/biology/lop6.pdf",
            "videos": []
        },
        # ... thêm nội dung cho các lớp khác ...
    }
}

class PDFPhysicsAssistant:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = {}  # Lưu trữ nội dung theo lớp
        self.embeddings = {  # Khởi tạo dictionary embeddings cho từng môn học
            'physics': {},
            'chemistry': {},
            'biology': {}
        }
        self.current_grade = None
        self.current_subject = None  # Thêm biến theo dõi môn học hiện tại
        self.context = """
        Bạn là trợ lý khoa học cho học sinh. Hãy giải thích các khái niệm một cách đơn giản, 
        dễ hiểu và phù hợp với trình độ của học sinh. Chỉ trả lời dựa trên tài liệu được cung cấp.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.initialize_documents()

    def initialize_documents(self):
        """Khởi tạo và load sẵn tất cả tài liệu"""
        for subject, grades in DOCUMENTS.items():
            for grade, info in grades.items():
                try:
                    self.load_pdf(info['pdf_path'], grade, subject)
                    self.logger.info(f"Loaded documents for grade {grade} in subject {subject}")
                except Exception as e:
                    self.logger.error(f"Error loading grade {grade} in subject {subject}: {str(e)}")

    def load_pdf(self, pdf_path: str, grade: str, subject: str, chunk_size: int = 1000):
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

                # Lưu trữ theo môn học và lớp
                if subject not in self.documents:
                    self.documents[subject] = {}
                self.documents[subject][grade] = chunks
                
                # Tạo embeddings
                if grade not in self.embeddings[subject]:
                    self.embeddings[subject][grade] = [self.model.encode(chunk) for chunk in chunks]

                return True
        except Exception as e:
            self.logger.error(f"Error loading PDF for grade {grade} in subject {subject}: {str(e)}")
            return False

    def set_current_grade(self, grade: str, subject: str):
        """Đặt lớp hiện tại để trả lời câu hỏi"""
        if subject in self.documents and grade in self.documents[subject]:
            self.current_grade = grade
            self.current_subject = subject
            return True
        return False

    def get_relevant_docs(self, question: str, top_k: int = 3) -> List[str]:
        """Tìm các đoạn văn bản liên quan từ tài liệu của lớp và môn học hiện tại"""
        if not self.current_grade or not self.current_subject:
            return []
            
        try:
            question_embedding = self.model.encode(question)
            if self.current_subject in self.embeddings and self.current_grade in self.embeddings[self.current_subject]:
                current_embeddings = self.embeddings[self.current_subject][self.current_grade]
                scores = [np.dot(question_embedding, doc_emb) for doc_emb in current_embeddings]
                top_indices = np.argsort(scores)[-top_k:]
                return [self.documents[self.current_subject][self.current_grade][i] for i in top_indices]
            return []
        except Exception as e:
            self.logger.error(f"Error in get_relevant_docs: {str(e)}")
            return []

    def get_answer(self, question: str) -> Optional[str]:
        """Trả lời câu hỏi dựa trên tài liệu của lớp và môn học hiện tại"""
        try:
            if not self.current_grade or not self.current_subject:
                return "Vui lòng chọn lớp và môn học trước khi đặt câu hỏi."
                
            relevant_docs = self.get_relevant_docs(question)
            if not relevant_docs:
                return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."
                
            client = openai.OpenAI(api_key=self.api_key)
            
            messages = [
                {"role": "system", "content": self.context},
                {"role": "system", "content": f"Đang trả lời cho học sinh lớp {self.current_grade} môn {self.current_subject}"},
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
    subject = data.get('subject', 'physics')  # mặc định là vật lý
    if subject in DOCUMENTS and grade in DOCUMENTS[subject]:
        assistant.set_current_grade(grade, subject)
        return jsonify({'message': f'Đã chọn lớp {grade} môn {subject}'})
    return jsonify({'error': 'Lớp hoặc môn học không hợp lệ'}), 400  

@app.route('/get_topics', methods=['POST'])
def get_topics():
    data = request.get_json()
    grade = data.get('grade')
    subject = data.get('subject', 'physics')
    if subject in DOCUMENTS and grade in DOCUMENTS[subject]:
        return jsonify(DOCUMENTS[subject][grade]['Lý thuyết'])
    return jsonify({'error': 'Lớp hoặc môn học không hợp lệ'}), 400

@app.route('/get_videos', methods=['POST'])
def get_videos():
    data = request.get_json()
    grade = data.get('grade')
    subject = data.get('subject', 'physics')
    if subject in DOCUMENTS and grade in DOCUMENTS[subject]:
        return jsonify(DOCUMENTS[subject][grade]['videos'])
    return jsonify({'error': 'Lớp hoặc môn học không hợp lệ'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'Chưa nhập câu hỏi'}), 400
    answer = assistant.get_answer(question)
    return jsonify({'answer': answer})

@app.route('/document')
def documents():
    # Chuyển đổi tên môn học sang tiếng Việt
    subject_names = {
        'physics': 'Vật lý',
        'chemistry': 'Hóa học',
        'biology': 'Sinh học'
    }
    
    # Tạo dữ liệu có tên môn học tiếng Việt
    formatted_documents = {}
    for subject, grades in DOCUMENTS.items():
        formatted_documents[subject_names.get(subject, subject)] = grades
        
    return render_template('document.html', documents=formatted_documents)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8085)