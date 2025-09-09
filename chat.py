from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings  # or google_genai embeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import GoogleGenerativeAI
from langchain.chains import LLMChain, RetrievalQA

# 1. Load PDFs
loader = UnstructuredPDFLoader("papers_dir/", mode="elements")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embeddings + Index
emb = GooglePalmEmbeddings()          # or google_genai embeddings wrapper
vectordb = FAISS.from_documents(chunks, emb)

# 4. Prompt-generator LLM: create an explicit prompt template
prompt_gen_llm = GoogleGenerativeAI(model="gemini-1.5-pro")  # example
prompt_template = """You are a prompt engineer. Given: {user_task} and {metadata},
output a detailed, explicit instruction template for a literature review that includes:
- audience, scope, structure, required citations format, depth, and any constraints.
Return JSON with fields: instruction, sections, citation_instructions."""
prompt_chain = LLMChain(llm=prompt_gen_llm, prompt=PromptTemplate.from_template(prompt_template))
generated_prompt = prompt_chain.run({"user_task":"Literature review", "metadata":"research papers on X"})

# 5. RAG / final generation
retriever = vectordb.as_retriever(search_kwargs={"k":8})
qa_chain = RetrievalQA.from_chain_type(
    llm=GoogleGenerativeAI(model="gemini-1.5-pro"),
    chain_type="stuff",  # or map_reduce/refine for long inputs
    retriever=retriever
)
final_review = qa_chain.run({"query": generated_prompt})
