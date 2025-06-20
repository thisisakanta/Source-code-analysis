from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from src.helper import load_embedding, repo_ingestion
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Setup Flask
app = Flask(__name__)

# Load API Key for OpenRouter (for DeepSeek)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY  # used by ChatOpenAI with base override

# === Load Embedding Model ===
embeddings = load_embedding()
persist_directory = "db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# === Setup DeepSeek LLM via OpenRouter ===
llm = ChatOpenAI(
    model_name="deepseek-ai/deepseek-coder:6.7b",
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# === Memory + Conversational Retrieval Chain ===
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
    memory=memory
)

# === Routes ===

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    if request.method == 'POST':
        repo_url = request.form['question']
        repo_ingestion(repo_url)
        os.system("python store_index.py")  # assumes it stores index into Chroma DB
        return jsonify({"response": f"Repo {repo_url} ingested and indexed."})
    return jsonify({"response": "Invalid method."})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if msg.strip().lower() == "clear":
        os.system("rm -rf repo")
        return "🧹 Repo folder cleared."
    
    result = qa(msg)
    print(result["answer"])
    return result["answer"]


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
