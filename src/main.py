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

def process_text_file(file_path, chunk_size, overlap):
    content = load_text_file(file_path)
    # print("no of characters: ", len(content))
    # print("first 300 characters: ", content[:300])
    # print("--------------- chunks ---------------")
    chunks = chunk_text(content, chunk_size, overlap)
    # print("chunk length: ", len(chunks))
    # for index, chunk in enumerate(chunks):
    #     print("\n--- chunk", index + 1, "---")
    #     print(chunk)
    # print("--------------- structured documents with metadata ---------------")
    documents = create_documents(chunks, file_path, 'text')
    return documents

def create_fake_embeddings(text):
    lower_text = text.lower()

    return [
        lower_text.count("python"),
        lower_text.count("programming")
    ]

def calculate_distance(embedding1, embedding2):
    distance = 0 
    for index in range(len(embedding1)):
        difference = embedding1[index] - embedding2[index]
        distance = distance + difference * difference
    return distance

def get_distance(document):
    return document['distance']

def search_documents(query, documents, top_k):
    query_embeddings = create_fake_embeddings(query)
    for document in documents:
        document["distance"] = calculate_distance(query_embeddings, document['embedding'])
    sorted_documents = sorted(documents, key=get_distance)
    return sorted_documents[:top_k]



def main():
    documents = process_text_file("data/sample.txt", 200, 50)
    for document in documents:
        document["embedding"] = create_fake_embeddings(document["text"])

    # create a query embeddings based on query input
    # compare the distances between query embeddings and document embeddings
    query = "what is Python Programming"

    results = search_documents(query, documents, 3)
    
    print("\n--- search results ---")
    for index, document in enumerate(results):
        print("\n--- result", index + 1, "---")
        print("distance:", document["distance"])
        print("metadata:", document["metadata"])
        print("text:", document["text"])
    

if __name__ == "__main__":
    main()

