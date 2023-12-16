import pandas as pd
import numpy as np
import re
import PyPDF2
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import os

# Function to read the contents of a PDF file
def read_pdf(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ''

        # Iterate through each page and extract text
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()

    return text

# Function to read all PDF files in a directory
def read_all_pdfs_concatenate(directory_path):
    long_text = ''
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            long_text += read_pdf(file_path) + "\n\n"  
    
    return long_text

# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
   
   qa_prompt = """Use the following pieces of information to answer the user's question.
   If you don't know the answer, just say that you don't know, don't try to make up an answer
   Context: {context}
   Question: {question}
   Only return the helpful answer below and nothing else.
   """
   prompt = PromptTemplate(template=qa_prompt,
                            input_variables=['context', 'question'])
   return prompt



def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':10}),
                                       return_source_documents=True)
    return dbqa

# Instantiate QA object
def setup_dbqa(llm,path):

   # Load embeddings for BiomedNLP-PubMedBERT model
    embeddings = HuggingFaceEmbeddings(model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                                   model_kwargs={'device': 'cpu'})

    vectordb = FAISS.load_local(path+'/models/vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa