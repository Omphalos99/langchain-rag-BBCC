import streamlit as st
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def extract_context_answer(text):
    start_index = text.find("context:") + len("context:")
    end_index = text.find("Answer", start_index)
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index].strip()
    else:
        return "context or Answer not found in the text"

def extract_shortened_books(filenames):
    shortened_names = []
    for filename in filenames:
        start_index = filename.find("books\\") + len("books\\")
        end_index = filename.find(".txt", start_index)
        if start_index != -1 and end_index != -1:
            shortened_names.append(filename[start_index:end_index])
        else:
            shortened_names.append("Invalid filename")
    return shortened_names
    
# Streamlit app
def app():
    st.title("Preguntas Básicas de Cultura Colombiana")

    # Input field for user query
    query_text = st.text_input("Escribe tu pregunta:")

    # Button to trigger the chatbot response
    if st.button("Preguntar"):
        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            st.write("No tengo idea, tal vez esto no existe en la Biblioteca.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        shortened_prompt=extract_context_answer(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        shortened_sources=extract_shortened_books(sources)
        formatted_response = f"Respuesta: {response_text}\n\nTextos:\n\n {shortened_prompt}\n\nFuentes: {shortened_sources}"
        st.write(formatted_response)

if __name__ == "__main__":
    app()
