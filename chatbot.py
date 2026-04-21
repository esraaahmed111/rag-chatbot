# Install
# pip install groq faiss-cpu sentence-transformers numpy matplotlib


# Imports
import os
import glob
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from groq import Groq


# API Key
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL   = "llama-3.3-70b-versatile"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq(api_key=GROQ_API_KEY)
print("Groq ready!")


# Load Text Files
FOLDER = './documents'

def load_text_files(folder_path):
    documents = []
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

    if not txt_files:
        print(f"No .txt files found in '{folder_path}'")
        return documents

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        if content:
            documents.append({'filename': filename, 'content': content})
            print(f"  Loaded: {filename} ({len(content):,} chars)")

    print(f"\n  Total files: {len(documents)}")
    return documents


if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

    with open(f'{FOLDER}/ai_basics.txt', 'w') as f:
        f.write("""Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine Learning is a subset of AI that allows systems to learn from data.
Deep Learning is a subset of Machine Learning that uses neural networks with many layers.
Natural Language Processing (NLP) is a branch of AI focused on understanding human language.
Computer Vision is a field of AI that enables machines to interpret and understand images.
Reinforcement Learning is a type of ML where an agent learns by interacting with an environment.
Transfer Learning allows models trained on one task to be applied to a different task.
BERT is a transformer model developed by Google for NLP tasks.
GPT is a generative model developed by OpenAI for text generation tasks.
RAG stands for Retrieval Augmented Generation which combines search with text generation.""")

    with open(f'{FOLDER}/python_basics.txt', 'w') as f:
        f.write("""Python is a high-level, interpreted programming language.
Python was created by Guido van Rossum and released in 1991.
Python is known for its simple and readable syntax.
Lists in Python are ordered, mutable collections defined with square brackets.
Dictionaries in Python store key-value pairs defined with curly braces.
Functions in Python are defined using the def keyword.
Classes in Python are defined using the class keyword and support OOP.
Pandas is a Python library for data manipulation and analysis.
NumPy is a Python library for numerical computing with arrays.
Scikit-learn is a Python library for machine learning algorithms.""")

    with open(f'{FOLDER}/nlp_topics.txt', 'w') as f:
        f.write("""Tokenization is the process of splitting text into individual words or tokens.
Stop words are common words like the, is, at that carry little meaning.
Lemmatization reduces words to their base form, for example running becomes run.
TF-IDF stands for Term Frequency Inverse Document Frequency.
Word embeddings represent words as dense numerical vectors.
Word2Vec is a model that learns word embeddings from large text corpora.
Sentiment Analysis determines whether text expresses positive, negative or neutral opinion.
Named Entity Recognition identifies entities like names, dates and locations in text.
Text summarization condenses long documents into shorter versions.
Machine Translation automatically translates text from one language to another.""")

    print("Demo files created!\n")

documents = load_text_files(FOLDER)


# Chunking
def split_into_chunks(documents, chunk_size=200, overlap=50):
    chunks = []
    for doc in documents:
        words = doc['content'].split()
        start = 0
        while start < len(words):
            end        = start + chunk_size
            chunk_text = ' '.join(words[start:end])
            chunks.append({
                'text'    : chunk_text,
                'source'  : doc['filename'],
                'chunk_id': len(chunks)
            })
            start += chunk_size - overlap
    return chunks

chunks = split_into_chunks(documents, chunk_size=200, overlap=50)
print(f"  Chunks created: {len(chunks)}")


# Embeddings + FAISS
print("\n  Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("  Model loaded!\n")

print("  Converting to vectors...")
chunk_texts      = [c['text'] for c in chunks]
chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
chunk_embeddings = np.array(chunk_embeddings).astype('float32')

VECTOR_DIM  = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
faiss_index.add(chunk_embeddings)
print(f"\n  FAISS ready — {faiss_index.ntotal} vectors stored")


# Retrieval
def retrieve_chunks(question, top_k=3):
    question_vec       = embedder.encode([question]).astype('float32')
    distances, indices = faiss_index.search(question_vec, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'text'    : chunks[idx]['text'],
            'source'  : chunks[idx]['source'],
            'distance': float(distances[0][i])
        })
    return results


# Generate Answer
def generate_answer(question, context_chunks):
    context = ""
    for chunk in context_chunks:
        context += f"[Source: {chunk['source']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have information about that in the provided documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.2,
        max_tokens  = 512
    )
    return response.choices[0].message.content.strip()


# Visualizations
def visualize_retrieval(question, top_k=5):
    results = retrieve_chunks(question, top_k=top_k)
    labels  = [f"{r['source']}\n\"{r['text'][:40]}...\"" for r in results]
    scores  = [1 / (1 + r['distance']) for r in results]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, scores, color='#3498db', edgecolor='black')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', va='center', fontsize=10)
    plt.xlabel('Similarity Score (higher = more relevant)')
    plt.title(f'Retrieved Chunks for: "{question}"', fontsize=12)
    plt.xlim(0, 1.15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

visualize_retrieval("What is deep learning?")


# Interactive Chat With Memory
chat_history = []

def chat(question, top_k=3):
    relevant_chunks = retrieve_chunks(question, top_k=top_k)

    context = ""
    for chunk in relevant_chunks:
        context += f"[Source: {chunk['source']}]\n{chunk['text']}\n\n"

    messages = [{
        "role"   : "system",
        "content": """You are a helpful assistant that answers questions
based only on the provided document context.
Be concise and accurate. If you don't know, say so."""
    }]

    for turn in chat_history:
        messages.append({"role": "user",      "content": turn['question']})
        messages.append({"role": "assistant", "content": turn['answer']})

    messages.append({
        "role"   : "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        temperature = 0.2,
        max_tokens  = 512
    )

    answer = response.choices[0].message.content.strip()
    chat_history.append({'question': question, 'answer': answer})
    return answer


# Interactive Loop
print("\n" + "="*60)
print("RAG Chatbot Ready! (Jupyter Version)")
print("   Type 'quit'    -> exit")
print("   Type 'history' -> see past Q&A")
print("   Type 'clear'   -> clear memory")
print("="*60)

while True:
    question = input("\nYou: ").strip()

    if question.lower() == 'quit':
        print("\nGoodbye!")
        break
    if question == '':
        print("Please type a question!")
        continue
    if question.lower() == 'history':
        if not chat_history:
            print("No history yet.")
        else:
            for i, turn in enumerate(chat_history):
                print(f"\n  Q{i+1}: {turn['question']}")
                print(f"  A{i+1}: {turn['answer'][:120]}...")
        continue
    if question.lower() == 'clear':
        chat_history.clear()
        print("Memory cleared!")
        continue

    answer = chat(question)
    print(f"\nBot: {answer}")
