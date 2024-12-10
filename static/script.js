let currentGrade = null;
let currentSubject = 'physics';

function formatMarkdown(text) {
    // Format bold text: **text** -> <strong>text</strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

function createMessageElement(type, content) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    
    // Xử lý format
    const prefix = type === 'question' ? 'Q: ' : 'A: ';
    const formattedContent = formatMarkdown(content);
    
    div.innerHTML = prefix + formattedContent;
    
    // Trigger MathJax to render any math formulas in the new content
    setTimeout(() => {
        MathJax.typesetPromise([div]).catch((err) => console.log('MathJax error:', err));
    }, 100);
    
    return div;
}

function selectGrade(grade) {
    // Reset active state
    document.querySelectorAll('.grade-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Find and activate clicked button
    const clickedButton = [...document.querySelectorAll('.grade-btn')]
        .find(btn => btn.textContent.includes(`Lớp ${grade}`));
    if (clickedButton) {
        clickedButton.classList.add('active');
    }

    // Set current grade
    currentGrade = grade;
    
    // Call API to set grade
    fetch('/set_grade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ grade: grade, subject: currentSubject })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            // Load topics and videos for selected grade
            loadTopics(grade);
            loadVideos(grade);
        }
    })
    .catch(error => {
        console.error('Error setting grade:', error);
        alert('Có lỗi xảy ra khi chọn lớp');
    });
}

function loadTopics(grade) {
    fetch('/get_topics', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ grade: grade, subject: currentSubject })
    })
    .then(response => response.json())
    .then(topics => {
        const topicsGrid = document.getElementById('topicsGrid');
        topicsGrid.innerHTML = '';
        topics.forEach(topic => {
            const topicDiv = document.createElement('div');
            topicDiv.className = 'topic-item';
            topicDiv.textContent = topic;
            topicsGrid.appendChild(topicDiv);
        });
    })
    .catch(error => {
        console.error('Error loading topics:', error);
        const topicsGrid = document.getElementById('topicsGrid');
        topicsGrid.innerHTML = '<div class="error">Lỗi khi tải chủ đề</div>';
    });
}

function loadVideos(grade) {
    fetch('/get_videos', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ grade: grade, subject: currentSubject })
    })
    .then(response => response.json())
    .then(videos => {
        const videoGrid = document.getElementById('videoGrid');
        videoGrid.innerHTML = '';
        videos.forEach(video => {
            const videoCard = document.createElement('div');
            videoCard.className = 'video-card';
            videoCard.innerHTML = `
                <h4>${video.title}</h4>
                <a href="${video.url}" target="_blank" rel="noopener noreferrer">
                    <i class="fas fa-play-circle"></i> Xem video
                </a>
            `;
            videoGrid.appendChild(videoCard);
        });
    })
    .catch(error => {
        console.error('Error loading videos:', error);
        const videoGrid = document.getElementById('videoGrid');
        videoGrid.innerHTML = '<div class="error">Lỗi khi tải video</div>';
    });
}

function askQuestion() {
    if (!currentGrade) {
        alert('Vui lòng chọn lớp trước khi đặt câu hỏi');
        return;
    }

    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    const chatHistory = document.getElementById('chatHistory');
    const loading = document.getElementById('loading');

    if (!question) return;

    // Add question to chat history
    const questionDiv = createMessageElement('question', question);
    chatHistory.appendChild(questionDiv);

    // Clear input
    questionInput.value = '';

    // Show loading
    loading.style.display = 'block';

    // Send question
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';

        if (data.error) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'message error';
            errorDiv.textContent = data.error;
            chatHistory.appendChild(errorDiv);
        } else {
            const answerDiv = createMessageElement('answer', data.answer);
            chatHistory.appendChild(answerDiv);
        }

        chatHistory.scrollTop = chatHistory.scrollHeight;
    })
    .catch(error => {
        loading.style.display = 'none';
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message error';
        errorDiv.textContent = 'Lỗi khi gửi câu hỏi';
        chatHistory.appendChild(errorDiv);
    });
}

// Add enter key support for question input
document.getElementById('questionInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion();
    }
});

function selectSubject(subject) {
    currentSubject = subject;
    // Cập nhật UI
    document.querySelectorAll('.subject-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.closest('.subject-btn').classList.add('active');
    document.getElementById('currentSubject').textContent = `Môn học: ${getSubjectName(subject)}`;
    // Reset topics và videos
    document.getElementById('topicsGrid').innerHTML = '';
    document.getElementById('videoGrid').innerHTML = '';
    document.getElementById('chatHistory').innerHTML = '';
}

function getSubjectName(subject) {
    const names = {
        'physics': 'Vật lý',
        'chemistry': 'Hóa học',
        'biology': 'Sinh học'
    };
    return names[subject] || subject;
}