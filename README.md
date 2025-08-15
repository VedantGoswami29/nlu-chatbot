# NLU FAQ Chatbot Server

A high-performance, easy-to-use FAQ chatbot server built with **Flask** and **Sentence Transformers**. This server provides a simple API endpoint to find the most semantically relevant answer to a user's query from a predefined set of questions and answers.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-2.2+-orange)

## 🌟 Features

-   **Fast & Asynchronous:** Built on FastAPI for high performance.
-   **State-of-the-Art NLU:** Uses Sentence Transformers for powerful semantic search, understanding the meaning behind words, not just keywords.
-   **Easy to Configure:** Manage data paths and model names through a simple `.env` file.
-   **Dynamic Data Loading:** Automatically loads and parses all `.txt` files from a specified FAQ directory on startup.
-   **CORS Ready:** Pre-configured with Cross-Origin Resource Sharing (CORS) middleware to allow requests from any origin, making frontend integration seamless.
-   **Simple API:** A clean and straightforward API with a single endpoint for asking questions.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   Python 3.8+
-   `pip` package manager

### 2. Clone the Repository

```bash
git clone https://github.com/VedantGoswami29/nlu-chatbot.git
cd nlu-chatbot
```

### 3. Set Up a Virtual Environment

#### For Unix/macOS
``` bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows
``` bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
``` bash
pip install -r requirements.txt
```
### 5. Prepare FAQ Data

The server loads data from .txt files located in the directory specified by FAQ_PATH in your .env file (defaults to faq/).

Create the faq directory and add one or more .txt files. The parser expects a specific format for each Q&A entry, separated by a line of underscores.

Example: faq/general.txt

``` plaintext
Question ID: GEN-001
Question: What are your business hours?
Answer: Our office is open from 9:00 AM to 6:00 PM, Monday to Friday.
Keywords: hours, open, timing, business hours

____________________________________

Question ID: GEN-002
Question: Where is your office located?
Answer: We are located at 123 Tech Park, Innovation Drive, Silicon Valley.
Keywords: location, address, find us, direction

____________________________________

Question ID: FIN-001
Question: How can I request a refund?
Answer: To request a refund, please email our support team at support@example.com with your order details. The refund process usually takes 5-7 business days.
Keywords: refund, money back, return policy
```

### 6. Run the Server
``` bash
python main.py
```
The server will be running at http://127.0.0.1:8000. On startup, it will load the model and process the FAQ files. You will see log messages indicating the progress.