from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_documents(chunks, source, source_type):
    documents = []
    for index, chunk in enumerate(chunks):
        document = {
            "text": chunk,
            "metadata": {
                "source": source,
                "source_type": source_type,
                "chunk_index": index,
            }
        }
        documents.append(document)
    return documents
    
def chunk_text(content, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks 

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Runs the text ingestion pipeline: load, chunk, and attach metadata.
def process_text_file(file_path, chunk_size, overlap):
    content = load_text_file(file_path)
    chunks = chunk_text(content, chunk_size, overlap)
    documents = create_documents(chunks, file_path, 'text')
    return documents

# Creates a semantic embedding vector for the input text.
def create_embeddings(text):
    embedding = model.encode(text)
    return embedding.tolist()

# Adds an embedding vector to each document.
def add_embeddings(documents):
    for document in documents:
        document["embedding"] = create_embeddings(document["text"])
    return documents
    
class ChromaDBStore:
    def __init__(self, path, collection_name):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
# {
#     "id": "sample.txt-0",
#     "document": "Python is a high-level programming language...",
#     "embedding": [0.02, -0.14, 0.77, ...],
#     "metadata": {
#         "source": "data/sample.txt",
#         "source_type": "text",
#         "chunk_index": 0
#     }
# }
    def add_documents(self, documents):
        documents = add_embeddings(documents)
        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for document in documents:
            metadata = document['metadata']
            document_id = metadata["source"] + "-" + str(metadata["chunk_index"])

            ids.append(document_id)
            texts.append(document['text'])
            metadatas.append(metadata)
            embeddings.append(document['embedding'])

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query, top_k, where=None):
        query_embeddings = create_embeddings(query)
        return self.format_results(self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k,
            where=where
        ))
    
    def format_results(self, results):
        formatted_results = []
        ids = results["ids"][0]
        texts = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for index in range(len(ids)):
            formatted_result = {
                "id": ids[index],
                "text": texts[index],
                "metadata": metadatas[index],
                "distance": distances[index]
            }

            formatted_results.append(formatted_result)
        return formatted_results
    

def ingest_documents(vector_store):
    documents = []
    documents.extend(process_text_file("data/python_features.txt", 200, 50))
    documents.extend(process_text_file("data/python_history.txt", 200, 50))
    vector_store.add_documents(documents)

def build_context(results):
    context = []
    for result in results:
        metadata = result['metadata']
        source = metadata['source']
        chunk_index = metadata['chunk_index']
        text = result['text']

        context_part = (
            f"[Source: {source}, Chunk: {chunk_index}]\n"
            f"{text}"
        )
        context.append(context_part)
    
    return "\n\n".join(context)

def filter_results_by_distance(results, max_distance):
    filtered_results = []
    for result in results:
        if result['distance'] <= max_distance:
            filtered_results.append(result)
    return filtered_results

def build_prompt(question, context):
    if not context:
        context = "No relevant information in context"
    prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided context."

When you answer, cite the source and chunk you used in this format:
(Source: source_path, Chunk: chunk_number)

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

def query_documents(vector_store, query, where=None):
    results = vector_store.search(query, 3, where=where)
    filtered_results = filter_results_by_distance(results, 1.0)
    context = build_context(filtered_results)
    prompt = build_prompt(query, context)
    print(prompt)


def main():
    chromedb_store = ChromaDBStore('chroma_db', 'rag_documents')
    ingest_documents(chromedb_store)
    query_documents(
        chromedb_store,
        "Who created Python?",
        where={"source": "data/python_history.txt"}
    )

    query_documents(
        chromedb_store,
        "What programming styles does Python support?",
        where={"source": "data/python_features.txt"}
    )

    query_documents(
        chromedb_store,
        "what is your name?",
        where={"source": "data/python_features.txt"}
    )
    

if __name__ == "__main__":
    main()
