from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

PDF_DIR = Path(__file__).resolve().parent.parent / "docs"
print("Looking for PDFs in:", PDF_DIR)

documents = []

for pdf_file in PDF_DIR.rglob("*.pdf"):
    print(f"Loading: {pdf_file.name}")
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    documents.extend(pages)

print(f"\nTotal pages loaded: {len(documents)}")
