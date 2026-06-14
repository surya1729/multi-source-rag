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
    print("no of characters: ", len(content))
    print("first 300 characters: ", content[:300])
    print("--------------- chunks ---------------")
    chunks = chunk_text(content, chunk_size, overlap)
    print("chunk length: ", len(chunks))
    for index, chunk in enumerate(chunks):
        print("\n--- chunk", index + 1, "---")
        print(chunk)
    print("--------------- structured documents with metadata ---------------")
    documents = create_documents(chunks, file_path, 'text')
    return documents

def main():
    documents = process_text_file("data/sample.txt", 200, 50)
    print("documents length: ", len(documents))
    for index, document in enumerate(documents):
        print("\n--- document", index + 1, "---")
        print(document)


if __name__ == "__main__":
    main()

