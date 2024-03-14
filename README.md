# Chat with PDF Using Groq

This application allows users to engage in a chat conversation based on the content of a provided PDF document. It utilizes Groq, a language model API, along with a Retrieval Augmented Generation (RAG) implementation to generate responses relevant to the conversation context.

## Features

- **PDF Upload**: Users can upload a PDF document.
- **Model Selection**: Choose from different language models for the conversation.
- **Chat Interface**: Engage in a chat conversation with the system.
- **Context-Aware Responses**: Responses are generated based on the context of the conversation and the content of the PDF.

## Setup

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Set up environment variables:
   - `GOOGLE_API_KEY`: Google API key for embedding generation.
   - `groq_api_key`: API key for accessing the Groq service.
4. Run the application using `streamlit run app.py`.

## How to Use

1. Upon running the application, you'll see a sidebar on the left.
2. Choose the desired language model from the dropdown list.
3. Upload a PDF document using the file uploader in the sidebar.
4. Start typing messages in the chat input box.
5. The system will respond based on the context of the conversation and the content of the PDF.
6. The chat history is displayed in the main window, showing messages from both the user and the system.

## Implementation Details

- The `get_vectorstore_from_pdf` function extracts text from the uploaded PDF, splits it into chunks, and creates a vector store using Google Generative AI embeddings.
- Context retrieval and conversational RAG chains are created using Groq for generating responses based on the conversation context.
- The conversation history and vector store are stored in the session state for maintaining context across interactions.

## Dependencies

- Streamlit: User interface library for creating interactive web applications.
- Groq: Language model API for natural language processing tasks.
- Langchain: Library for various text processing tasks, including document loading, vectorization, and conversational AI.
