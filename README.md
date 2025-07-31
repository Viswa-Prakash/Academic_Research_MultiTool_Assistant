#  Academic Research Multi-Tool ReAct Agent

A conversational AI agent for academic research and scholarly questions, built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). The agent analyzes a user query, reasons step-by-step, and uses multiple specialized tools to provide authoritative, referenced, and useful answers for students, researchers, and lifelong learners.

---

##  Features

- **Research Paper Search:** Fetches, summarizes, and cites cutting-edge academic papers using Arxiv and Semantic Scholar.
- **Wikipedia & Web Lookup:** Retrieves encyclopedic and outbound info for background and literature reviews.
- **Translation Support:** Can translate scientific findings into other languages (if translation tool enabled).
- **Python REPL:** Runs calculations/statistics/math for polled data, figures, and results inside papers.
- **Intelligent Reasoning:** Follows the ReAct workflow: Thought → Action → Observation → Final Answer.
- **Professional Summaries:** Presents referenced, readable summaries (with sources cited).
- **Streamlit Interface:** Simple, web-based UI.

---

##  Setup
### 1. Clone this Repository

```bash
git clone https://github.com/Viswa-Prakash/Academic_Research_MultiTool_Assistant.git
cd Academic_Research_MultiTool_Assistant

### 2. Install Dependencies

```bash
pip install -r requirements.txt

### 3. Configure Environment Variables

Create a `.env` file and add your API keys for the following services:

OPENAI_API_KEY=sk-xxxxxxx
SERPER_API_KEY=serper_xxxxxxx 



##  Usage


1. Start the agent web app:
```bash
streamlit run app.py
```

2. Enter your query in the text box and click "Ask Agent".
Example queries:
Summarize the most cited papers in reinforcement learning from the last two years. Also, translate the main findings into Spanish.
Find Wikipedia and Semantic Scholar info on 'transformers in language modeling', and calculate the average of reported accuracies [98, 95, 97].
Show top Arxiv papers on graph neural networks and give a short summary with references.

3. Get a clear, final, referenced answer only.
---