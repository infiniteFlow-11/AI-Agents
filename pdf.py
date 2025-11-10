from PyPDF2 import PdfReader
def get_pdf_text(pdf_files):
    raw_text=""
    for pdf in pdf_files:         
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

# print(get_pdf_text("./seo-ebook.pdf"))
pdf_files = ['./seo-ebook.pdf']
print(get_pdf_text(pdf_files))