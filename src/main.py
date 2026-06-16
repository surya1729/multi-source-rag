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

def get_distance(document):
    return document['distance']

# Searches documents using squared Euclidean distance.
# Lower distance means higher similarity.
def search_documents(query, documents, top_k):
    query_embeddings = create_embeddings(query)
    for document in documents:
        document["distance"] = squared_euclidean_distance(query_embeddings, document['embedding'])
    sorted_documents = sorted(documents, key=get_distance)
    return sorted_documents[:top_k]

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

def get_score(document):
    return document['score']

def get_manhattan_distance(document):
    return document['manhattan_distance']

# Searches documents using cosine similarity.
# Higher cosine similarity means higher similarity.
def search_documents_by_cosine(query, documents, top_k):
    query_embeddings = create_embeddings(query)
    for document in documents:
        document['score'] = cosine_similarity(query_embeddings, document['embedding'])
    sorted_documents = sorted(documents, key=get_score, reverse=True)
    return sorted_documents[:top_k]

# Calculates dot product between two embeddings.
def dot_product(embedding1, embedding2):
    score = 0
    for index in range (len(embedding1)):
        score = score + embedding1[index] * embedding2[index]
    return score

# Searches documents using dot product.
# Higher dot product means higher similarity.
def search_documents_by_dot_product(query, documents, top_k):
    query_embeddings = create_embeddings(query)
    for document in documents:
        document['score'] = dot_product(query_embeddings, document['embedding'])
    sorted_document = sorted(documents, key=get_score, reverse=True)
    return sorted_document[:top_k]

# Calculates Manhattan distance by summing absolute differences.
def manhattan_distance(embedding1, embedding2):
    score = 0
    for index in range (len(embedding1)):
        score = score + abs(embedding1[index]-embedding2[index])
    return score

# Searches documents using Manhattan distance.
# Lower Manhattan distance means higher similarity.
def search_documents_by_manhattan_distance(query, documents, top_k):
    query_embeddings = create_embeddings(query)
    for document in documents:
        document['manhattan_distance'] = manhattan_distance(query_embeddings, document['embedding'])
    sorted_distance = sorted(documents, key=get_manhattan_distance)
    return sorted_distance[:top_k]


def main():
    documents = process_text_file("data/sample.txt", 200, 50)
    documents = add_embeddings(documents)
    query = "Who created Python?"
    results = search_documents(query, documents, 3)
    print("\n--- search results ---")
    for index, document in enumerate(results):
        print("\n--- result based on euclidean distance", index + 1, "---")
        print("distance:", document["distance"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    results = search_documents_by_cosine(query, documents, 3)
    for index, document in enumerate(results):
        print("\n--- result based on cosine similarity", index + 1, "---")
        print("score:", document["score"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    results = search_documents_by_dot_product(query, documents, 3)
    for index, document in enumerate(results):
        print("\n--- result based on dot product", index + 1, "---")
        print("score:", document["score"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    results = search_documents_by_manhattan_distance(query, documents, 3)
    for index, document in enumerate(results):
        print("\n--- result based on manhattan", index + 1, "---")
        print("manhattan_distance:", document["manhattan_distance"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    

if __name__ == "__main__":
    main()
