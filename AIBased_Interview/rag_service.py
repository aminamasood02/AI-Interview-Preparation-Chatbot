import os
import pickle
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from django.conf import settings
from .models import QADocument
import google.generativeai as genai

class RAGService:
    def __init__(self):
        self.model_name = 'all-MiniLM-L6-v2'
        self.embedder = None
        self.index = None
        self.documents = []
        self.index_path = os.path.join(settings.BASE_DIR, 'rag_data')
        self.ensure_rag_directory()
        
    def ensure_rag_directory(self):
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
    
    def initialize_embedder(self):
        if self.embedder is None:
            print('Loading sentence transformer model...')
            self.embedder = SentenceTransformer(self.model_name)
            print('Model loaded successfully!')
    
    def load_qa_data_from_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f'Loaded CSV with {len(df)} rows')
            
            qa_pairs = []
            for idx, row in df.iterrows():
                qna_text = str(row['qna']).strip()
                if not qna_text or qna_text == 'nan':
                    continue
                
                if '?' in qna_text:
                    parts = qna_text.split('?', 1)
                    if len(parts) == 2:
                        question = parts[0].strip() + '?'
                        answer = parts[1].strip()
                        
                        if question and answer:
                            qa_pairs.append({
                                'question': question,
                                'answer': answer,
                                'combined_text': f'Question: {question} Answer: {answer}'
                            })
            
            print(f'Extracted {len(qa_pairs)} valid Q&A pairs')
            return qa_pairs
            
        except Exception as e:
            print(f'Error loading CSV: {e}')
            return []
    
    def save_qa_to_database(self, qa_pairs):
        print('Saving Q&A pairs to database...')
        QADocument.objects.all().delete()
        
        documents = []
        for idx, qa in enumerate(qa_pairs):
            doc = QADocument.objects.create(
                question=qa['question'],
                answer=qa['answer'],
                combined_text=qa['combined_text'],
                embedding_index=idx
            )
            documents.append(doc)
        
        print(f'Saved {len(documents)} documents to database')
        return documents
    
    def create_embeddings(self, qa_pairs):
        self.initialize_embedder()
        print('Creating embeddings...')
        texts = [qa['combined_text'] for qa in qa_pairs]
        
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        print(f'Created embeddings with shape: {embeddings.shape}')
        return embeddings
    
    def create_faiss_index(self, embeddings):
        print('Creating FAISS index...')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        print(f'Created FAISS index with {index.ntotal} vectors')
        return index
    
    def save_index(self, index, embeddings, documents):
        print('Saving FAISS index and data...')
        faiss.write_index(index, os.path.join(self.index_path, 'faiss_index.bin'))
        np.save(os.path.join(self.index_path, 'embeddings.npy'), embeddings)
        
        doc_data = [{'id': doc.id, 'question': doc.question, 'answer': doc.answer} for doc in documents]
        with open(os.path.join(self.index_path, 'documents.pkl'), 'wb') as f:
            pickle.dump(doc_data, f)
        
        print('Index and data saved successfully!')
    
    def load_index(self):
        try:
            print('Loading FAISS index and data...')
            index_file = os.path.join(self.index_path, 'faiss_index.bin')
            if not os.path.exists(index_file):
                print('No existing index found. Please build the index first.')
                return False
            
            self.index = faiss.read_index(index_file)
            
            with open(os.path.join(self.index_path, 'documents.pkl'), 'rb') as f:
                self.documents = pickle.load(f)
            
            self.initialize_embedder()
            print(f'Loaded index with {len(self.documents)} documents')
            return True
            
        except Exception as e:
            print(f'Error loading index: {e}')
            return False
    
    def search_similar_documents(self, query, k=5):
        if self.index is None or self.embedder is None:
            if not self.load_index():
                return []
        
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'document': doc,
                    'similarity_score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def generate_response_with_context(self, query, similar_docs):
        try:
            api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            context = "\n\n".join([
                f"Context {i+1}:\nQ: {doc['document']['question']}\nA: {doc['document']['answer']}"
                for i, doc in enumerate(similar_docs[:3])
            ])
            
            prompt = f"""You are an intelligent interview preparation assistant. Based on the provided context, answer the user's question comprehensively and accurately.

Context Information:
{context}

User Question: {query}

Instructions:
1. Use the context information to provide a detailed and accurate answer
2. If the context directly answers the question, use that information
3. If the context is related but doesn't directly answer, use it as supporting information
4. Provide examples or explanations where helpful
5. Keep the answer focused and relevant to interview preparation
6. If the context isn't sufficient, provide a general answer but mention the limitation

Answer:"""

            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f'Error generating response: {e}')
            return 'I apologize, but I am having trouble generating a response at the moment. Please try again.'
    
    def process_query(self, query, user=None):
        start_time = time.time()
        
        try:
            similar_docs = self.search_similar_documents(query, k=5)
            
            if not similar_docs:
                response = "I couldn't find relevant information for your query. Please try rephrasing your question."
                processing_time = time.time() - start_time
                return {
                    'query': query,
                    'response': response,
                    'similar_documents': [],
                    'processing_time': processing_time
                }
            
            response = self.generate_response_with_context(query, similar_docs)
            processing_time = time.time() - start_time
            
            result = {
                'query': query,
                'response': response,
                'similar_documents': similar_docs,
                'processing_time': processing_time
            }
            
            if user:
                from .models import RAGQuery
                RAGQuery.objects.create(
                    user=user,
                    query=query,
                    response=response,
                    relevant_documents=[doc['document']['id'] for doc in similar_docs],
                    similarity_scores=[doc['similarity_score'] for doc in similar_docs],
                    processing_time=processing_time
                )
            
            return result
            
        except Exception as e:
            print(f'Error processing query: {e}')
            return {
                'query': query,
                'response': f'Error processing your query: {str(e)}',
                'similar_documents': [],
                'processing_time': time.time() - start_time
            }
    
    def build_index_from_csv(self, csv_path):
        print('Starting RAG index building process...')
        
        qa_pairs = self.load_qa_data_from_csv(csv_path)
        if not qa_pairs:
            print('No valid Q&A pairs found!')
            return False
        
        documents = self.save_qa_to_database(qa_pairs)
        embeddings = self.create_embeddings(qa_pairs)
        index = self.create_faiss_index(embeddings)
        self.save_index(index, embeddings, documents)
        
        print('RAG index building completed successfully!')
        return True

rag_service = RAGService()
