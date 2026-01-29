from PyPDF2 import PdfReader


def pdf_to_chunks(pdf_path: str, chunk_size: int = 1000, overlap: int = 100):
    """
    Liest ein PDF ein und teilt den Text in Chunks auf.

    :param pdf_path: Pfad zur PDF-Datei
    :param chunk_size: Länge eines Chunks (in Zeichen)
    :param overlap: Überlappung zwischen Chunks (in Zeichen)
    :return: Liste von Text-Chunks
    """
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = []
    start = 0
    text_length = len(full_text)

    while start < text_length:
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


chunks = pdf_to_chunks("sample.pdf")
print(chunks)
