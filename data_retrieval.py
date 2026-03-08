import os
from groq import Groq
from tavily import TavilyClient
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(name=os.getenv("INDEX_NAME"), host=os.getenv("HOST_NAME"))
def generate_embeddings(texts):
    return model.encode(texts)

def get_response_from_lama(prompt: str, model: str = "openai/gpt-oss-20b") -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Groq API call failed: {e}")

def get_answer_from_query(user_query):
  template = """Analyze the following Question and give one word answer of YES if the question is about pakistan history or about the mughal empire,
  about the events that occured in indian subcontinent after mughal empire and before partition of the subcontinent and formation of Pakistan.
   answer with NO otherwise. Question: {query}"""

  prompt = PromptTemplate(input_variables=["query"],template=template)
  final_prompt = prompt.format(query=user_query)
  response = get_response_from_lama(final_prompt)
  if response == "NO":
    return ["Query was not About Pakistan History","Error"]

  query_embedding = generate_embeddings([user_query])[0]
  query_result = index.query(vector=query_embedding.tolist(), top_k=3)
  if query_result['matches'][0]['score']>0.65:
    context = "\n".join([" ".join(match['values']) for match in query_result['matches']])
    reply_from = 'Book'
  else:
    reply = tavily_client.search(user_query)
    relevant_content = [i["content"] for i in reply["results"] if i.get("score",0) > 0.65]
    context = "\n\n".join(relevant_content)
    reply_from = 'Internet Search'

  llama_prompt = f"""Context:\n{context}\n\nUser Query: {user_query}\n
  Restrictions:Don't over-format the result. Don't add <br> tokens. Don't add info other than the context.\n
  Answer:"""
  response = get_response_from_lama(llama_prompt)
  return response,reply_from