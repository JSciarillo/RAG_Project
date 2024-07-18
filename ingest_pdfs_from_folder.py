import os
import ollama
import chromadb
import PyPDF2

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the page is not empty
                text.append(page_text)
    return text

# Specify the path to the folder containing PDF files
folder_path = "C:\\Users\\jasmi\\Downloads\\Python_Projects\\AI_Retrieval_Augmented_Generation_Model\\PDF_Folder"

# Initialize documents list
documents = []

# Iterate over all PDF files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, filename)
        documents.extend(extract_text_from_pdf(pdf_path))

# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Store each document in a vector embedding database
for i, d in enumerate(documents):
    response = ollama.embeddings(model="nomic-embed-text", prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

# An example prompt
prompt = "What is the Equinix Private AI?"

# Generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="nomic-embed-text"
)
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]

# Generate a response combining the prompt and data we retrieved
output = ollama.generate(
    model="llama3",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])
