# DocChatLLM

DocChatLLM is a tool designed to facilitate chat-like dialogues with uploaded documents. This application allows users to interact with their documents in a conversational manner, extracting information and getting answers to specific queries.

## Features

- Upload documents in various formats (e.g., PDF, Image-Documents, DOCX).
- Engage in chat-like interactions to extract information from documents.
- User-friendly interface powered by Streamlit.

## Installation

To install and run DocChatLLM, follow these steps:

### Prerequisites

- Python 3.10 or higher
- `pip`
- Virtual environment tool (e.g. `venv` or `virtualenv`)

### Step-by-Step Installation

1. Clone repository

```bash
git clone https://github.com/naetherm/docchatllm.git
cd DocChatLLM

```

2. Create a virtual environment

Using `venv`

```bash
python -m venv venv
```

Using `virtualenv`

```bash
virtualenv venv
```

3. Activate the environment

- On Windows: `venv\Scripts\activate`
- On MacOS and Linux: `source venv/bin/activate`

4. Install requred packages

```bash
pip install -r requirements.txt
```

## Usage

After installing the required packages, you can start the application with the following command:

```bash
streamlit run app.py
```
