# PDF_QA: Chat with Your PDFs (with Table Support)

A modern web app to upload PDF documents and chat with them using advanced AI. Supports text and table extraction, multi-turn conversational context, and a beautiful dark UI inspired by GPT/Claude.

---

## 🚀 Features

- **Upload PDFs** (up to 50MB)
- **Chat with your documents**: Ask questions about the content, get detailed answers
- **Table extraction**: Extracts tables and makes them available for Q&A
- **Multi-turn context**: The assistant remembers your previous questions for coherent, topic-aware conversations
- **Modern dark UI**: Responsive, beautiful, and easy to use
- **Predefined prompts**: Quick-start questions always available above the chat box
- **File management**: List, select, and delete uploaded PDFs

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, pdfplumber, sentence-transformers, numpy, ollama
- **Frontend**: HTML, CSS, JavaScript (no framework, pure and fast)
- **AI Model**: [Ollama](https://ollama.com/) (default: mistral:latest, can be changed)

---

## ⚡ Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/pdf_qa.git
cd pdf_qa
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the backend
```bash
python main.py
```
- The backend runs on `http://localhost:5000`
- Make sure [Ollama](https://ollama.com/) is running locally (default model: mistral:latest)

### 4. Open the frontend
- Open `NewUI.html` in your browser (double-click or use a local server)
- The app will connect to the backend at `localhost:5000`

---

## 🧑‍💻 API Endpoints

### `POST /api/upload`
Upload a PDF file. Returns file ID and stats.

### `POST /api/question`
Ask a question about a PDF. Body:
```json
{
  "question": "...",
  "file_id": "...",
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
Returns: `{ "answer": "..." }`

### `GET /api/files`
List all uploaded PDFs.

### `DELETE /api/files/<file_id>`
Delete a PDF and its processed data.

### `GET /api/sample_prompts`
Get the list of predefined prompts for the UI.

---

## 📄 Usage Tips
- **Tables**: The assistant can answer questions about tables (e.g., "Summarize the financial data table on page 2").
- **Multi-turn chat**: Ask follow-up questions for deeper insights—the assistant remembers the conversation.
- **Predefined prompts**: Use the chips above the chat box for quick-start questions.
- **File management**: Delete PDFs you no longer need from the sidebar.

---

## 🤝 Contributing
Pull requests welcome! Please open an issue to discuss major changes.

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [sentence-transformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- [Mistral AI](https://mistral.ai/)

---

Enjoy chatting with your PDFs! 