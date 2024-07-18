import ollama
import chromadb
import PyPDF2

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text())
    return text

# Specify the path to your PDF file
pdf_path = "C:\\Users\\jasmi\\Downloads\\Python_Projects\\AI_Retrieval_Augmented_Generation_Model\\PDF_Folder\\WEF_The_Global_Cooperation_Barometer_2024.pdf"
documents = extract_text_from_pdf(pdf_path)

# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Store each document in a vector embedding database
for i, d in enumerate(documents):
    if d:  # Ensure the page is not empty
        response = ollama.embeddings(model="nomic-embed-text", prompt=d)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[d]
        )

# An example prompt
prompt = "Create a summary of this text."

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
