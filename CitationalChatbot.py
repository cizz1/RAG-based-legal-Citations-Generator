import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from PyPDF2 import PdfMerger, PdfReader
from datetime import datetime
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="shashaaa/fine_tune_embeddnew_SIH_2")

st.set_page_config(page_title="Citational RAG Chatbot", layout="wide")
st.title("Citational RAG chatbot")
# st.write("Upload pdfs")

# api_key=st.text_input("enter your groq api key:",type="password")
api_key = os.getenv("GROQ_API_KEY")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    if 'store' not in st.session_state:
        st.session_state.store={}

    session_id=st.text_input("Unique case name",placeholder="An unique session identifier")
    tm = datetime.now()
    if session_id in st.session_state.store:
        fname = st.session_state.store[session_id][1]
    else:
        fname = f"Case: {session_id}  Time: {tm.strftime('%d-%m-%Y %H:%M:%S')}"
    st.write(fname)
    
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Choose a PDF",type="pdf",accept_multiple_files=True)
    merger = PdfMerger()
    # print(uploaded_files)
    merger.append('citataion_special.PDF')

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./tmp/temp-{session_id}.pdf"
            with open(temppdf,"wb") as file:
                file.write(b'')
            #     file_name=uploaded_file.name
            merger.append(PdfReader(uploaded_file))
        merger.write(temppdf)
        merger.close()


        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
        # print(documents)

        
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits= text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits , embedding=embeddings)
        retriever = vectorstore.as_retriever()


        contextualize_q_system_prompt=(
            "given a chat history and latest user question"
            "which might refrence context in chat history"
            "formulate a standalone question which can be understood"
            "without thta chat history.DO NOt answer the question"
            "just refromulate it if needed and otherwise return it as is"
        )

        contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_system_prompt)


        system_prompt=(
            #"You are a Virtual Lawyer, an AI model extensively trained on the Indian judicial dataset. "
            #"Your task is to provide predictive analysis and possible outcomes of current legal cases based on past history and interactions."
            # "You assist users with understanding legal concepts, answering legal questions, and providing guidance on legal matters. "
            # "You do not provide legal advice but offer information to help users make informed decisions. "
            # "Maintain a neutral tone and avoid taking sides, even when there are emotional or ethical factors involved. "
            # "Always provide relevant legal context, including references to applicable acts, laws, or past cases. "
            # "If you don't know the answer, say that you don't know. "
            # "Keep your responses informative but not overly lengthy."
            # "You are a highly knowledgeable legal assistant specializing in generating accurate legal citations based on the Bluebook citation style."
            # "Your task is to assist users by providing precise and correctly formatted legal citations for various legal documents, cases, statutes, and other sources. "
            # "Follow these guidelines:"

            # "1.Citation Style: Always use the Bluebook citation style."
            # "2. Accuracy: Ensure all citations are accurate and reflect the most current legal standards."
            # "3. Formatting: Pay close attention to the formatting rules of the Bluebook, including punctuation, capitalization, and italicization."
            # "4. Contextual Understanding: Understand the context of the user’s query to provide the most relevant citations."
            # "5. Examples: Provide examples of correctly formatted citations when necessary to illustrate the proper format."
            # "6. User Interaction: Be polite, professional, and clear in your responses. If the user provides incomplete information, ask clarifying questions to ensure accuracy."
            # "7. Limitations: If you are unable to generate a citation due to insufficient information or other limitations, inform the user politely and suggest alternative ways to obtain the needed information."
            "You are a highly knowledgeable legal assistant AI, trained on extensive datasets of court cases from previous years."
            " Your primary task is to generate accurate legal citations in compliance with the JCLG Guide to Bluebook 20th edition. When responding to user queries, you must: " 
            "1. Retrieve Relevant Information: Use the provided context and your training data to find the most relevant court cases and legal precedents."
            "2. Generate Citations: Format the citations according to the JCLG Guide to Bluebook 20th edition. "
            "Ensure all citations are precise and adhere to the correct format."
            "3. Provide Contextual Responses: Along with citations, offer brief explanations or summaries of the cases when necessary to enhance user understanding. "
            "4. Acknowledge Limitations: If the information is not available or the query cannot be answered based on the provided data, politely inform the user."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )


        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->ChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=[ChatMessageHistory(), fname]
            return st.session_state.store[session_id][0]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        st.sidebar.header("Ask Your Question")
        user_input= st.text_input("your question", value='')
        if user_input:
            session_history=get_session_history(session_id)
            response =  conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )

            st.write("Assitant:",response['answer'])
            st.write(st.session_state.store[session_id])
            st.write("chat_history:",session_history.messages)

else:
    st.warning("please enter api key")