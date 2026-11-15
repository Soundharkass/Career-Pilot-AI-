# CareerPilot AI

A local AI career chatbot powered by the Google Gemini API. It retrieves relevant passages from a local career guide or your uploaded files, explains technology concepts, explores tech careers, practices interviews, and creates week-by-week learning plans with a local Python tool.

## Features

- Gemini chat using Google's `google-genai` SDK and Interactions API.
- Streams response text into the chat as Gemini generates it.
- Configurable approximate input-token budget, output-token limit, and low-latency generation settings.
- Retrieval-augmented answers from local guides and uploads, using fast keyword matching by default or optional Gemini embeddings.
- Upload PDF, DOCX, TXT, Markdown, and CSV files in the chat and ask questions about their contents.
- Retrieved sources are cited by filename and section; add or edit Markdown guides to tailor the knowledge base.
- A local `build_learning_plan` tool that Gemini can call during a conversation.
- Multi-turn chat context retained in the browser tab.
- Gemini API key stays on the Python server in `.env`; it is never sent to browser JavaScript.
- Clear diagnostic errors are shown in the chat and server terminal.

This version does not use live web search. Its answers can retrieve passages from the local guide, but that guide does not provide current job listings, salaries, or news. The app must not claim that it checked those sources. Gemini API free-tier access and request limits depend on model and account. Check current availability and quotas in [Google AI Studio](https://aistudio.google.com/).

Uploaded files are stored in `knowledge_base/uploads/` on this computer. With semantic retrieval enabled, the first question after adding or changing a document sends its extracted text to Gemini to create embeddings; relevant passages and your chat are sent to Gemini to answer each question. Scanned PDFs without selectable text and password-protected PDFs are not supported. Remove uploaded files from the document list in the app.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys)
- Internet access while the app is running

Google AI Studio offers free-tier access to selected Gemini models within their usage limits. The API key itself does not guarantee access to every model or unlimited usage.

## Setup on Windows PowerShell

1. Open PowerShell in this project folder.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this in the current PowerShell window, then activate again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Copy the example settings file and edit it:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Paste your Gemini API key after `GEMINI_API_KEY=`. Keep the key private; `.env` is excluded by `.gitignore`.

5. Start the app:

   ```powershell
   .\.venv\Scripts\python.exe careerbot.py
   ```

6. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

To stop the server, return to PowerShell and press **Ctrl+C**.

## Example prompts

- “I want to become a junior cybersecurity analyst. I am a beginner and can study 5 hours per week. Make me a 6-week roadmap.”
- “Explain APIs like I’m new to programming, then give me a tiny practice task.”
- “Give me one junior developer interview question at a time and coach me after each answer.”
- “Compare software testing and data analysis as entry-level careers.”

## Project structure

```text
CareerPilotAI/
├── careerbot.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── app.js
│   └── styles.css
└── knowledge_base/
    └── career_guide.md
```

## Notes

- The API key is read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) by the Python server.
- `GEMINI_EMBEDDING_MODEL` selects the embedding model; it defaults to `gemini-embedding-2`. The first semantic retrieval after changing the embedding model rebuilds the local embedding cache.
- `GEMINI_MAX_INPUT_TOKENS` sets an approximate input budget (default `6000`); the app trims older chat context to fit without making an extra token-count request.
- `GEMINI_MAX_OUTPUT_TOKENS` sets the combined limit for model thinking and response text (default `2048` in the app). `GEMINI_THINKING_LEVEL=low` reduces latency for Gemini 3.8 Flash.
- `GEMINI_FAST_RETRIEVAL=true` uses local keyword retrieval to avoid an embedding request during each reply. Set it to `false` to use semantic embedding retrieval.
- Add `.md` files to `knowledge_base/` to give the assistant more reference material. The app rebuilds its ignored `.rag_index.json` cache when those files change.
- The app accepts PDF, DOCX, TXT, Markdown, and CSV uploads up to 16 MB each, with a maximum of 25 uploaded files.
- Conversation history currently lives in browser memory and clears when the page reloads.
- This is a local learning project. Add login, request limits, persistent storage, and production hosting controls before exposing it publicly.
- Free-tier usage limits and model access may change; see Google's current [Gemini API billing guide](https://ai.google.dev/gemini-api/docs/billing).
