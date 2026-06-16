from sentence_transformers import SentenceTransformer

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

# Calculates squared Euclidean distance between two embeddings.
def squared_euclidean_distance(embedding1, embedding2):
    distance = 0 
    for index in range(len(embedding1)):
        difference = embedding1[index] - embedding2[index]
        distance = distance + difference * difference
    return distance

# Calculates cosine similarity using dot product and vector lengths.
# cosine_similarity = dot_product / (embedding1_length * embedding2_length)
def cosine_similarity(embedding1, embedding2):
    dot_product = 0
    embedding1_length = 0
    embedding2_length = 0
    for index in range(len(embedding1)):
        dot_product = dot_product + embedding1[index] * embedding2[index]
        embedding1_length = embedding1_length + embedding1[index] * embedding1[index]
        embedding2_length = embedding2_length + embedding2[index] * embedding2[index]
    
    embedding1_length = embedding1_length ** 0.5
    embedding2_length = embedding2_length ** 0.5

    return dot_product/(embedding1_length * embedding2_length)

# Calculates dot product between two embeddings.
def dot_product(embedding1, embedding2):
    score = 0
    for index in range (len(embedding1)):
        score = score + embedding1[index] * embedding2[index]
    return score

# Calculates Manhattan distance by summing absolute differences.
def manhattan_distance(embedding1, embedding2):
    score = 0
    for index in range (len(embedding1)):
        score = score + abs(embedding1[index]-embedding2[index])
    return score

def search_documents(query, documents, top_k, score_function, score_name, higher_is_better):
    query_embedding = create_embeddings(query)

    for document in documents:
        document[score_name] = score_function(query_embedding, document["embedding"])

    sorted_documents = sorted(
        documents,
        key=lambda document: document[score_name],
        reverse=higher_is_better
    )

    return sorted_documents[:top_k]


def main():
    documents = process_text_file("data/sample.txt", 200, 50)
    documents = add_embeddings(documents)
    query = "Who created Python?"
    euclidean_results = search_documents(
        query,
        documents,
        3,
        squared_euclidean_distance,
        "squared_euclidean_distance",
        False
    )
    print("\n--- search euclidean_results ---")
    for index, document in enumerate(euclidean_results):
        print("\n--- result based on euclidean distance", index + 1, "---")
        print("squared_euclidean_distance:", document["squared_euclidean_distance"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])

    cosine_similarity_results = search_documents(
        query,
        documents,
        3,
        cosine_similarity,
        "cosine_similarity",
        True,
    )
    for index, document in enumerate(cosine_similarity_results):
        print("\n--- result based on cosine similarity", index + 1, "---")
        print("cosine_similarity:", document["cosine_similarity"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    
    dot_product_results = search_documents(
        query,
        documents,
        3,
        dot_product,
        "dot_product",
        True
    )
    for index, document in enumerate(dot_product_results):
        print("\n--- result based on dot product", index + 1, "---")
        print("dot_product:", document["dot_product"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])

    manhattan_distance_results = search_documents(
        query,
        documents,
        3,
        manhattan_distance,
        "manhattan_distance",
        False
    )
    for index, document in enumerate(manhattan_distance_results):
        print("\n--- result based on manhattan", index + 1, "---")
        print("manhattan_distance:", document["manhattan_distance"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    

if __name__ == "__main__":
    main()
