import json
import pandas as pd
import torch
import flask
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session  # Add this import
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import os
import gc  # For garbage collection
import numpy as np
import tempfile
import uuid
import subprocess
from pathlib import Path
from transformers import pipeline
import faiss
from typing import List, Dict
import textwrap
from openai import OpenAI
from datetime import timedelta
import time
import pickle  # Add this import
import re
from peft import PeftModel, PeftConfig


# Flexible import for OpenAI
try:
    from openai import OpenAI

    client = OpenAI(
        api_key="YOUR API-KEY HERE"
    )
except ImportError:
    import openai

    client = openai.api_requestor.APIRequestor()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.urandom(24)  # Secret key for session management

# Path to fine-tuned model - update this to your actual model location
FINETUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "phi2-lora-finetuned")


# Update the session configuration
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=5)  # Sessions last 5 hours
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")
Session(app)

# Make sure the session directory exists
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)


DEFAULT_TO_API = True  # Global flag to control default model


# Initialize global variables
llm_model = None
llm_tokenizer = None
embedder = None
faiss_index = None
document_store = []
device = None
use_api_model = DEFAULT_TO_API  # Default to using the API
model_load_error = None


# Document manager for persistent storage between requests
class DocumentManager:
    def __init__(self):
        self.sessions = {}
        self.storage_path = os.path.join(os.getcwd(), "document_store")
        os.makedirs(self.storage_path, exist_ok=True)
        self.load_from_disk()  # Load existing data when initialized

    def store_document(self, session_id, abstract, chunks, embeddings=None):
        self.sessions[session_id] = {
            "abstract": abstract,
            "chunks": chunks,
            "embeddings": embeddings,
            "timestamp": time.time(),
            "metadata": {},
        }
        self.save_to_disk()  # Save after every update

    def store_metadata(self, session_id, metadata):
        """Store additional metadata for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["metadata"] = metadata
            self.save_to_disk()

    def get_metadata(self, session_id):
        """Get metadata for a session"""
        if session_id in self.sessions:
            return self.sessions[session_id].get("metadata", {})
        return {}

    def get_document(self, session_id):
        return self.sessions.get(session_id)

    def clean_old_sessions(self, max_age=3600):  # 1 hour
        current_time = time.time()
        to_remove = []
        for session_id, data in self.sessions.items():
            if current_time - data["timestamp"] > max_age:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]

        if to_remove:
            self.save_to_disk()

    def get_current_chunks(self, session_id):
        """Get chunks for the current session"""
        session_data = self.get_document(session_id)
        if session_data:
            return session_data["chunks"]
        return []

    def save_to_disk(self):
        """Save sessions data to disk"""
        try:
            # We'll save everything except the embeddings which can be regenerated
            save_data = {}
            for session_id, data in self.sessions.items():
                save_data[session_id] = {
                    "abstract": data["abstract"],
                    "chunks": data["chunks"],
                    "timestamp": data["timestamp"],
                    "metadata": data.get("metadata", {}),
                }

            with open(os.path.join(self.storage_path, "sessions.pkl"), "wb") as f:
                pickle.dump(save_data, f)
            print("Document store saved to disk")
        except Exception as e:
            print(f"Error saving document store: {e}")

    def load_from_disk(self):
        """Load sessions data from disk without trying to regenerate embeddings"""
        try:
            file_path = os.path.join(self.storage_path, "sessions.pkl")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    saved_data = pickle.load(f)

                # Just load the data without embeddings
                for session_id, data in saved_data.items():
                    self.sessions[session_id] = data
                    # Remove any embeddings to avoid issues
                    if "embeddings" in self.sessions[session_id]:
                        del self.sessions[session_id]["embeddings"]

                print(f"Loaded {len(saved_data)} sessions from disk")
        except Exception as e:
            print(f"Error loading document store: {e}")

    def ensure_embeddings(self, session_id):
        """Ensure embeddings exist for a session"""
        if embedder is None:
            print("Warning: Embedder not initialized, can't generate embeddings")
            return False

        session_data = self.get_document(session_id)
        if (
            session_data
            and "chunks" in session_data
            and ("embeddings" not in session_data or session_data["embeddings"] is None)
        ):
            try:
                chunks = session_data["chunks"]
                embeddings = embedder.encode(chunks, convert_to_tensor=True)
                self.sessions[session_id]["embeddings"] = embeddings.cpu().numpy()
                return True
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                return False
        return True


# Initialize the document manager
doc_manager = DocumentManager()

# Constants for RAG
CHUNK_SIZE = 384
CHUNK_OVERLAP = 64
EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2
MAX_LENGTH = 2048  # for response generation


def clear_gpu_memory():
    """Function to clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_cache_size():
    """Get the size of the Hugging Face cache in GB"""
    import os
    import shutil

    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        size = shutil.disk_usage(cache_dir).used / (1024**3)  # Convert to GB
        return size
    return 0


def clear_model_cache():
    """Clear model cache to free up space"""
    from huggingface_hub import delete_cache_folder
    import shutil
    import os

    # Clear HF cache
    try:
        delete_cache_folder()
        print("Hugging Face cache cleared")
    except Exception as e:
        print(f"Error clearing HF cache: {e}")

    # Clear custom cache
    try:
        if os.path.exists(".model_cache"):
            shutil.rmtree(".model_cache")
            print("Custom model cache cleared")
    except Exception as e:
        print(f"Error clearing custom cache: {e}")

    # Clear PyTorch cache
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("PyTorch cache cleared")
    except Exception as e:
        print(f"Error clearing PyTorch cache: {e}")


def load_model():
    global llm_model, llm_tokenizer, embedder, faiss_index, device, model_load_error

    try:
        # Set device to CPU explicitly
        device = torch.device("cpu")
        print("Forcing CPU usage for all models")

        # Use a smaller, more CPU-friendly model
        base_model_name = "microsoft/phi-2"
        print(f"Loading base model {base_model_name} and LoRA adapter from {FINETUNED_MODEL_PATH}")

        # Initialize tokenizer and model
        llm_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Ensure pad token is set properly
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print("Set pad_token to eos_token")

        # Load the base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load the LoRA adapter on top of the base model
        llm_model = PeftModel.from_pretrained(
            base_model, 
            FINETUNED_MODEL_PATH,
            torch_dtype=torch.float32
        )

        print("LoRA adapter loaded successfully")

        # Load sentence transformer for embeddings
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        # Initialize FAISS index
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)

        print("Models loaded successfully on CPU!")
        return True

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback

        traceback.print_exc()
        model_load_error = str(e)

        return False


def chunk_text(text: str, min_chunk_size: int = 100) -> List[str]:
    """Enhanced text chunking with better boundary detection and abstract separation"""
    chunks = []

    # Split by double newlines which often separate abstracts
    abstract_parts = text.split("\n\n")

    for abstract_part in abstract_parts:
        # Skip empty parts
        if not abstract_part.strip():
            continue

        # Preprocess text to normalize spacing within each abstract
        abstract_part = " ".join(abstract_part.split())

        start = 0
        while start < len(abstract_part):
            end = start + CHUNK_SIZE

            if end >= len(abstract_part):
                chunks.append(abstract_part[start:])
                break

            # Look for sentence boundaries
            potential_end = end
            while potential_end > start + min_chunk_size:
                if (
                    potential_end < len(abstract_part)
                    and abstract_part[potential_end - 1] in {".", "!", "?"}
                    and (
                        potential_end >= len(abstract_part)
                        or abstract_part[potential_end].isupper()
                        or abstract_part[potential_end].isspace()
                    )
                ):
                    end = potential_end
                    break
                potential_end -= 1

            chunks.append(abstract_part[start:end])
            start = end - CHUNK_OVERLAP

    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > min_chunk_size]


def process_abstract_for_rag(abstract: str, session_id: str):
    """Process abstract and store in RAG system"""
    global faiss_index

    try:
        # Create a new FAISS index for this session
        session_faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)

        # Chunk the abstract
        chunks = chunk_text(abstract)
        print(f"Created {len(chunks)} chunks from the abstract")

        if not chunks:
            print("Warning: No chunks were created from the abstract")
            return False

        # Compute embeddings for chunks
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        chunk_embeddings_np = chunk_embeddings.cpu().numpy()

        # Add to FAISS index
        session_faiss_index.add(chunk_embeddings_np)

        # Store in document manager
        doc_manager.store_document(
            session_id=session_id,
            abstract=abstract,
            chunks=chunks,
            embeddings=chunk_embeddings_np,
        )

        print(
            f"Successfully processed abstract: added {len(chunks)} chunks for session {session_id}"
        )
        return True

    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def retrieve_relevant_context(
    question: str, session_id: str, top_k: int = 5, threshold: float = 0.25
) -> List[str]:
    """Improved context retrieval with similarity threshold and better handling of multiple abstracts"""
    try:
        # Get chunks for this session
        chunks = doc_manager.get_current_chunks(session_id)

        if not chunks:
            print(f"No chunks found for session {session_id}")
            return ["No context available."]

        # Create FAISS index for these chunks
        temp_faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        temp_faiss_index.add(chunk_embeddings.cpu().numpy())

        # Encode question
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        question_embedding_np = question_embedding.cpu().numpy().reshape(1, -1)

        # Search in FAISS index - increase top_k to get more potential matches
        D, I = temp_faiss_index.search(
            question_embedding_np, min(top_k + 5, len(chunks))
        )

        # Filter by similarity threshold and remove duplicates
        relevant_chunks = []
        seen_content = set()

        for dist, idx in zip(D[0], I[0]):
            if idx >= len(chunks):
                print(
                    f"Warning: Index {idx} out of bounds for chunks of length {len(chunks)}"
                )
                continue

            chunk = chunks[idx]
            # Convert distance to similarity score (FAISS returns L2 distance)
            similarity = 1 / (1 + dist)

            if similarity < threshold:
                continue

            # Remove near-duplicate chunks
            chunk_content = " ".join(chunk.split())
            if chunk_content not in seen_content:
                relevant_chunks.append(chunk)
                seen_content.add(chunk_content)

            if len(relevant_chunks) >= top_k:
                break

        print(
            f"Retrieved {len(relevant_chunks)} relevant chunks for question: {question}"
        )

        # If we have too few chunks, lower the threshold and try again
        if len(relevant_chunks) < 2 and threshold > 0.15:
            print("Too few chunks retrieved, lowering threshold...")
            return retrieve_relevant_context(
                question, session_id, top_k, threshold - 0.1
            )

        return relevant_chunks

    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        import traceback

        traceback.print_exc()
        return []


def generate_answer_with_api(question: str, context: List[str]) -> str:
    """Generate answer using API calls with improved handling of multiple abstract contexts"""
    try:
        # Validate inputs
        if not context:
            return "No context available to generate an answer."

        # Sort contexts by relevance to question
        context_scores = []
        question_embedding = embedder.encode(question, convert_to_tensor=True).to(
            device
        )

        for chunk in context:
            chunk_embedding = embedder.encode(chunk, convert_to_tensor=True).to(device)
            score = util.pytorch_cos_sim(question_embedding, chunk_embedding)[0][0]
            context_scores.append((chunk, score))

        # Sort contexts by relevance score
        sorted_contexts = [
            chunk
            for chunk, _ in sorted(context_scores, key=lambda x: x[1], reverse=True)
        ]

        # Combine contexts with markers to help model distinguish between different abstracts
        combined_context = ""
        for i, ctx in enumerate(
            sorted_contexts[:5]
        ):  # Limit to top 5 most relevant chunks
            combined_context += f"[Data_Chunk {i+1}]: {ctx}\n\n"

        # Create prompt with better instructions for handling multiple abstracts
        prompt = f"""Answer the following question based on the provided context from multiple scientific abstracts:

Context:
{combined_context}

Question: {question}

Instructions:
1. Base your answer solely on the information in the provided context
2. If different abstracts contain contradictory information, mention this in your answer
3. If the exact information isn't in any of the abstracts, say so clearly
4. Be concise but complete
5. If relevant information appears in multiple abstracts, synthesize it into a coherent answer
"""

        # Initialize AI client
        base_url = "https://api.deepseek.com"
        api_key = "YOUR API-KEY HERE"
        model = "deepseek-chat"

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Call the API with increased max_tokens to handle more complex answers
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context from scientific abstracts. Your answers should be accurate, comprehensive, and well-structured.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,  # Increased to handle more complex synthesis
            temperature=0.3,
            stream=False,
        )

        # Extract the answer
        answer = response.choices[0].message.content

        return answer

    except Exception as e:
        print(f"Error generating answer with API: {str(e)}")
        import traceback

        traceback.print_exc()
        return "I apologize, but I encountered an error while processing your question. Please try again."


# def generate_answer(question: str, context: List[str]) -> str:
#     """Simplified answer generation using TinyLlama for better CPU performance"""
#     try:
#         # Validate inputs
#         if not context:
#             return "No context available to generate an answer."

#         print(f"Generating answer with local model for question: {question}")
#         print(f"Available context chunks: {len(context)}")

#         # Use only the most relevant context to keep prompt short
#         # This helps with performance on CPU
#         context_text = " ".join(context[:2])  # Use only top 2 chunks

#         # Create a simple prompt for TinyLlama
#         prompt = f"You are a helpful AI assistant that answers questions about scientific papers on High Entropy Alloys.Base your answer solely on the information in the provided context. If the value asked is missing from the context for specific conditions then acknowledge the Unavailability of the Data and say so clearly.\n\nContext: {context_text}\n\nQuestion: {question}\n\nAnswer:"


#         print("Encoding prompt...")
#         input_ids = llm_tokenizer.encode(prompt, return_tensors="pt").to(device)

#         # Generate with minimal parameters for better CPU performance
#         print("Starting generation...")
#         with torch.no_grad():
#             outputs = llm_model.generate(
#                 input_ids=input_ids,
#                 max_new_tokens=50,  # Keep this small for faster generation
#                 do_sample=True,  # Use sampling for more natural text
#                 temperature=0.7,  # Lower temperature for more focused responses
#                 top_p=0.9,  # Nucleus sampling
#                 pad_token_id=llm_tokenizer.pad_token_id,
#                 eos_token_id=llm_tokenizer.eos_token_id,
#             )

#         print("Generation completed successfully")

#         # Decode the generated text
#         generated_text = llm_tokenizer.decode(
#             outputs[0][input_ids.shape[1] :], skip_special_tokens=True
#         )
#         print(f"Generated text: {generated_text}")

#         # If the answer is too short, provide a default response
#         if not generated_text or len(generated_text.split()) < 5:
#             return "Based on the provided context, I couldn't find specific information about the compression yield strength of the Nb20Cr20Mo10Ta10Ti20Zr20 alloy at 296K."

#         return generated_text.strip()

#     except Exception as e:
#         print(f"Detailed Error in generate_answer: {str(e)}")
#         import traceback

#         traceback.print_exc()
#         return "I apologize, but I encountered an error while processing your question. Please try again."

def generate_answer(question: str, context: List[str]) -> str:
    """Answer generation using fine-tuned Phi-2 model"""
    try:
        # Validate inputs
        if not context:
            return "No context available to generate an answer."

        print(f"Generating answer with fine-tuned model for question: {question}")
        print(f"Available context chunks: {len(context)}")

        # Use only the most relevant context to keep prompt short
        context_text = " ".join(context[:2])  # Use only top 2 chunks

        # Create prompt in the format expected by the fine-tuned model
        prompt = f"{context_text}\nQuestion: {question}\nAnswer:"

        print("Encoding prompt...")
        input_ids = llm_tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate with parameters matching your Kaggle notebook
        print("Starting generation...")
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids=input_ids,
                max_new_tokens=50,        # Match your Kaggle notebook
                do_sample=False,          # Deterministic output
                temperature=0.0,          # Exact match to your Kaggle settings
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
            )

        print("Generation completed successfully")

        # Decode the generated text
        generated_text = llm_tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"Generated text: {generated_text}")

        # If the answer is too short, provide a default response
        if not generated_text or len(generated_text.split()) < 5:
            return "Based on the provided context, I couldn't find specific information to answer your question."

        # return generated_text.strip()
        return generated_text.split('\n')[0]


    except Exception as e:
        print(f"Detailed Error in generate_answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return "I apologize, but I encountered an error while processing your question. Please try again."



def generate_answer_with_retrieval(
    question: str, session_id: str, top_k: int = 5, threshold: float = 0.25
) -> str:
    """Main function to generate answer using RAG with either local model or API"""
    try:
        # Get chunks for this session
        chunks = doc_manager.get_current_chunks(session_id)

        if not chunks:
            print(f"No chunks found for session {session_id}")
            return ["No context available."]

        # Ensure embeddings exist for this session
        doc_manager.ensure_embeddings(session_id)

        # Create FAISS index for these chunks
        temp_faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        temp_faiss_index.add(chunk_embeddings.cpu().numpy())

        # Validate if document store is populated for this session
        session_data = doc_manager.get_document(session_id)
        if not session_data or not session_data.get("chunks"):
            return "No documents have been processed. Please submit an abstract first."

        # Get relevant context for this session
        relevant_chunks = retrieve_relevant_context(question, session_id)

        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question. The abstract might not contain the specific details you're asking about."

        # Choose between API and local model
        if use_api_model:
            print(
                f"Retrieved {len(relevant_chunks)} relevant chunks. Generating answer with API..."
            )
            answer = generate_answer_with_api(question, relevant_chunks)
            print("API answer generation completed")
        else:
            print(
                f"Retrieved {len(relevant_chunks)} relevant chunks. Generating answer with local model..."
            )
            answer = generate_answer(question, relevant_chunks)
            print("Local model answer generation completed")

        return answer

    except Exception as e:
        print(f"Detailed Error in generate_answer_with_retrieval: {str(e)}")
        import traceback

        traceback.print_exc()
        return "I apologize, but I encountered an error while processing your question. Please try again."


# Function to process abstract using GPT API
def process_abstract_with_gpt(abstract_input):
    """Process abstract using GPT API to extract structured data about High Entropy Alloys"""

    # Determine if we're processing multiple abstracts
    abstracts = []
    if isinstance(abstract_input, list):
        abstracts = abstract_input
    elif isinstance(abstract_input, str):
        # Check if this is a combined abstract with separators
        if "\n\n---\n\n" in abstract_input:
            abstracts = abstract_input.split("\n\n---\n\n")
        else:
            # Single abstract
            abstracts = [abstract_input]
    else:
        # Not a string or list, handle as appropriate
        abstracts = [abstract_input]

    # List to collect results from all abstracts
    all_abstract_results = []
    all_table_data = []

    # Process each abstract individually
    for idx, abstract in enumerate(abstracts):
        try:
            # Generate the prompt for GPT - exactly as before
            prompt = f"""Your task is to process individual excerpts from materials science papers specifically focusing on High Entropy Alloys (HEAs). Only include High Entropy Alloys and exclude any Baseline Alloys from the output. Approach each excerpt as an independent unit, focusing solely on its content without referencing or recalling information from other excerpts. Convert these excerpts into a structured JSON format.

You'll encounter various data types in these texts, such as chemical compositions and performance properties, often accompanied by specific measurement conditions. Pay special attention to the 'compositions.' Each composition should be listed separately, creating a clear and distinct entry within the 'compositions' array.

For alloys where the composition contains a range of values (e.g., "Al": "x (0, 0.2, 0.4, 0.6, 0.8, 1.0)"), generate separate entries for each value of the composition, forming distinct alloy names and their corresponding composition. For example, for "AlxNb0.5TiV2Zr0.5", generate entries for each composition like "Al0.2Nb0.5TiV2Zr0.5", "Al0.4Nb0.5TiV2Zr0.5", and so on. Each entry should have its own 'name', 'composition', and respective 'performance_properties'.

For each composition, provide detailed information in subsections named 'performance_properties.' When dealing with numerical data, such as measurements or values, ensure these are included in a 'value' field and specify their corresponding units in a 'unit' field. Only include properties where the temperature at which the property was measured is explicitly mentioned. For properties that have multiple measurements at different temperatures, group them under the same property name, each with its corresponding value and temperature. Add a 'type' field to indicate the property type (e.g., 'hardness', 'strength').

Additionally, include the alloy name and the DOI of the paper in the JSON structure.

Include new fields to capture 'Equilibrium Conditions,' 'Single/Multiphase Material,' and 'Type of Phase Present.' Add this information under a dedicated 'material_conditions' section.

The output should be in the following structured JSON format:

{{
    "alloys": [
        {{
            "name": "Al0.5Mo0.5NbTa0.5TiZr",
            "composition": {{
                "Al": 0.5,
                "Mo": 0.5,
                "Nb": 1,
                "Ta": 0.5,
                "Ti": 1,
                "Zr": 1
            }},
            "performance_properties": {{
                "hardness": {{
                    "type": "hardness",
                    "measurements": [
                        {{
                            "temperature": "<Temperature1 with Unit>",
                            "value": "<Value1>",
                            "unit": "<Unit>"
                        }},
                        {{
                            "temperature": "<Temperature2 with Unit>",
                            "value": "<Value2>",
                            "unit": "<Unit>"
                        }}
                    ]
                }},
                "strength": {{
                    "type": "strength",
                    "measurements": [
                        {{
                            "temperature": "<Temperature1 with Unit>",
                            "value": "<Value1>",
                            "unit": "<Unit>"
                        }},
                        {{
                            "temperature": "<Temperature2 with Unit>",
                            "value": "<Value2>",
                            "unit": "<Unit>"
                        }}
                    ]
                }},
                ...
            }},
            "material_conditions": {{
                "equilibrium_conditions": "<Details>",
                "single_or_multiphase": "<Single/Multiphase>",
                "phase_type": "<Type of Phase>"
            }},
            "doi": "<DOI>"
        }},
        ...
    ]
}}

Process the following excerpt:

{abstract}
"""

            # Call the OpenAI API
            if isinstance(client, OpenAI):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts properties of High Entropy Alloys (HEAs) from scientific abstracts."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.3
                )
                json_output = response.choices[0].message.content.strip()
            else:
                response = client.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts properties of High Entropy Alloys (HEAs) from scientific abstracts."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.3
                )
                json_output = response["choices"][0]["message"]["content"].strip()

            # Clean and extract the JSON
            def clean_json_string(json_str):
                """Clean and prepare JSON string for parsing."""
                # Remove code block markers if present
                json_str = re.sub(r'```json\n|\n```', '', json_str)
                
                # Try to find the main JSON object
                try:
                    stack = []
                    start = json_str.find('{')
                    if start == -1:
                        return None
                    
                    for i in range(start, len(json_str)):
                        if json_str[i] == '{':
                            stack.append(i)
                        elif json_str[i] == '}':
                            stack.pop()
                            if not stack:  # We've found the matching closing brace
                                end = i + 1
                                return json_str[start:end]
                except:
                    return None
                
                return None

            def fix_json_formatting(json_str):
                """Fix common JSON formatting issues."""
                # Quote unquoted keys
                json_str = re.sub(r'([{,]\s*)(\w+)(:)', r'\1"\2"\3', json_str)
                
                # Fix trailing commas
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                # Fix missing commas between elements
                json_str = re.sub(r'([}\]"\'\d])(\s*[{\["\'])', r'\1,\2', json_str)
                
                return json_str

            # Clean and extract the JSON
            cleaned_json = clean_json_string(json_output)
            if not cleaned_json:
                print(f"Warning: Could not extract valid JSON from response for abstract {idx}")
                continue
                
            try:
                # Fix formatting issues
                cleaned_json = fix_json_formatting(cleaned_json)
                alloy_data = json.loads(cleaned_json)
                
                # Create a result entry with abstract and index
                abstract_result = {
                    "alloys": [],
                    "abstract": abstract,
                    "index": idx
                }
                
                # Filter and clean alloy data
                def is_numeric_composition(composition):
                    """Check if all values in the composition are numeric."""
                    try:
                        return all(isinstance(float(str(value)), float) for value in composition.values())
                    except (ValueError, TypeError):
                        return False

                def clean_alloy_name(name):
                    """Remove trailing commas from alloy names."""
                    if isinstance(name, str):
                        return name.rstrip(',')
                    return name

                def clean_value_unit(measurements):
                    """Remove trailing commas from 'value' and 'unit' keys in measurements."""
                    for measurement in measurements:
                        if 'value' in measurement and isinstance(measurement['value'], str):
                            measurement['value'] = measurement['value'].rstrip(',')
                        if 'unit' in measurement and isinstance(measurement['unit'], str):
                            measurement['unit'] = measurement['unit'].rstrip(',')

                def clean_material_conditions(material_conditions):
                    """Remove trailing commas from all string values in 'material_conditions'."""
                    for key, value in material_conditions.items():
                        if isinstance(value, str):
                            material_conditions[key] = value.rstrip(',')
                
                # Filter and clean the data
                valid_alloys = []
                if 'alloys' in alloy_data:
                    for alloy in alloy_data['alloys']:
                        # Clean the alloy name by removing trailing commas
                        alloy['name'] = clean_alloy_name(alloy['name'])
                        
                        # Check if the composition is numeric
                        if 'composition' in alloy and is_numeric_composition(alloy['composition']):
                            # Clean 'value' and 'unit' keys in performance_properties
                            if 'performance_properties' in alloy:
                                for property_name, property_data in alloy['performance_properties'].items():
                                    if 'measurements' in property_data:
                                        clean_value_unit(property_data['measurements'])
                            
                            # Clean 'material_conditions' section if present
                            if 'material_conditions' in alloy:
                                clean_material_conditions(alloy['material_conditions'])
                            
                            valid_alloys.append(alloy)
                
                # Update the abstract result with valid alloys
                abstract_result["alloys"] = valid_alloys
                
                # Add the entry to the results list if it contains valid alloys
                if valid_alloys:
                    all_abstract_results.append(abstract_result)
                
                # Generate table data for this abstract
                abstract_table_data = []
                
                def is_valid_temperature(temperature):
                    """Check if the temperature value is valid."""
                    valid_aliases = ["room temperature", "ambient", "Ambient", "RT", "Room Temperature", "room-temperature"]
                    
                    # Convert input to string to handle different types
                    temp_str = str(temperature).strip()
                    
                    # Check if it's a valid alias
                    if temp_str in valid_aliases:
                        return True
                    
                    # Check if temperature has numeric value and a valid unit
                    return bool(re.match(r"^\d+\s*[°C°FK]*$", temp_str))
                
                def is_valid_value(value):
                    """Check if the value is valid and not zero or equivalent."""
                    try:
                        # Attempt to convert to float
                        numeric_value = float(str(value).strip())
                        # Skip if value is 0 or 0.0
                        return numeric_value != 0.0
                    except (ValueError, TypeError):
                        # If conversion fails, return False
                        return False
                
                # Process each alloy for table data
                for alloy in valid_alloys:
                    alloy_name = alloy["name"]
                    doi = alloy.get("doi", "Not provided")
                    performance_properties = alloy.get("performance_properties", {})
                    material_conditions = alloy.get("material_conditions", {})
                    
                    # Process performance properties and their measurements
                    for property_name, property_data in performance_properties.items():
                        for measurement in property_data.get("measurements", []):
                            value = measurement.get("value", "N/A")
                            unit = measurement.get("unit", "N/A")
                            temperature = measurement.get("temperature", "N/A")
                            
                            # Validate the temperature
                            if not is_valid_temperature(temperature):
                                temperature = "Not specified"
                            
                            # Skip invalid or zero-equivalent values
                            if not is_valid_value(value):
                                continue
                            
                            # Skip rows where the value contains alphabets
                            if isinstance(value, str) and any(char.isalpha() for char in value):
                                continue
                            
                            # Add valid entry to table_data with the abstract index
                            abstract_table_data.append({
                                "Index": idx if idx > 0 else 0,
                                "DOI": doi,
                                "Alloy Name": alloy_name,
                                "Property Name": property_name.replace('_', ' ').title(),
                                "Property Type": property_data.get("type", property_name),
                                "Temperature": temperature,
                                "Value": value,
                                "Unit": unit,
                                "Equilibrium Conditions": material_conditions.get("equilibrium_conditions", "Not mentioned"),
                                "Single/Multiphase": material_conditions.get("single_or_multiphase", "Not mentioned"),
                                "Phase Type": material_conditions.get("phase_type", "Not specified")
                            })
                
                # Add this abstract's table data to the combined table data
                all_table_data.extend(abstract_table_data)
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error for abstract {idx}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing abstract {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue with next abstract
    
    # If we didn't get any successful results
    if not all_abstract_results:
        return {"success": False, "error": "No valid alloys found in any abstract"}
    
    # Return the results in the new format
    return {
        "success": True,
        "json_data": all_abstract_results,
        "table_data": all_table_data,
    }



# Add these functions to your app.py file
def run_qa_pair_generation(json_data, session_id):
    """Run the QA pair generation script with the given JSON data"""
    try:
        # Create a temporary file for the input
        input_file = os.path.join(app.config["SESSION_FILE_DIR"], f"{session_id}_input.json")
        
        # Write the JSON data to a file
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        
        # Create output directory
        output_dir = os.path.join(app.config["SESSION_FILE_DIR"], f"{session_id}_qa_pairs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the QA pair generation script
        cmd = [
            "python", 
            "qa_pair_gen.py", 
            input_file, 
            "--output_dir", output_dir,
            "--rating_threshold", "6"
        ]
        
        # Run the command (non-blocking)
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Store the paths in the session for later retrieval
        session["qa_pair_process"] = {
            "session_id": session_id,
            "input_file": input_file,
            "output_dir": output_dir,
            "status": "running"
        }
        
        print(f"Started QA pair generation for session {session_id}")
        return True
    except Exception as e:
        print(f"Error running QA pair generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_qa_pair_status(session_id):
    """Check if QA pair generation has completed"""
    try:
        qa_process = session.get("qa_pair_process", {})
        if not qa_process or qa_process.get("session_id") != session_id:
            return {"status": "not_started"}
        
        output_dir = qa_process.get("output_dir")
        final_output_file = os.path.join(output_dir, "final_qa_pairs.json")
        
        if os.path.exists(final_output_file):
            # Process completed
            return {"status": "completed", "output_file": final_output_file}
        else:
            # Still processing
            return {"status": "running"}
    except Exception as e:
        print(f"Error checking QA pair status: {str(e)}")
        return {"status": "error", "error": str(e)}

def load_qa_pairs(file_path):
    """Load QA pairs from the output file and convert to table format"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            qa_pairs_data = json.load(f)
        
        # Convert to tabular format
        table_data = []
        
        # Iterate through all contexts
        for context_key, context_data in qa_pairs_data.items():
            # Iterate through each alloy
            for alloy_name, categories in context_data.items():
                # Iterate through each category
                for category, qa_list in categories.items():
                    # Add each QA pair to the table
                    for qa in qa_list:
                        table_data.append({
                            "Context": context_key,
                            "Alloy": alloy_name.replace('"', '').strip(),
                            "Category": category,
                            "Question": qa["question"],
                            "Answer": qa["answer"]
                        })
        
        return {"success": True, "table_data": table_data}
    except Exception as e:
        print(f"Error loading QA pairs: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}




# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/submit-csv", methods=["POST"])
def submit_csv():
    try:
        # Check if file was uploaded
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"})

        file = request.files["file"]

        # Check if file is empty
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"})

        # Check if file is a CSV
        if not file.filename.endswith(".csv"):
            return jsonify({"success": False, "error": "File must be a CSV"})

        # Create a temporary file to store the uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Read the CSV file
            df = pd.read_csv(temp_file_path)

            # Check if the CSV has the expected columns
            if "Abstract" not in df.columns:
                return jsonify(
                    {"success": False, "error": "CSV must contain an 'Abstract' column"}
                )

            # Filter out empty abstracts
            df = df[df["Abstract"].notna() & (df["Abstract"] != "")]

            if len(df) == 0:
                return jsonify(
                    {"success": False, "error": "No valid abstracts found in the CSV"}
                )

            # Add DOI information if available
            has_doi = "DOI" in df.columns

            # Process each abstract individually and then combine
            all_abstracts = []
            for idx, row in df.iterrows():
                abstract = row["Abstract"]
                if has_doi and pd.notna(row["DOI"]):
                    abstract += f"\nDOI: {row['DOI']}"
                all_abstracts.append(abstract)

            # Combine all abstracts with clear separation
            combined_abstract = "\n\n---\n\n".join(all_abstracts)

            # Generate a session ID
            session_id = str(uuid.uuid4())

            # Process for RAG
            rag_success = process_abstract_for_rag(combined_abstract, session_id)
            if not rag_success:
                return jsonify(
                    {"success": False, "error": "Failed to process abstracts for RAG"}
                )

            # Process with GPT
            result = process_abstract_with_gpt(combined_abstract)
            # result = process_abstract_with_gpt(all_abstracts)


            if result["success"]:
                # Store in session
                session["current_session_id"] = session_id
                session["table_data"] = result["table_data"]
                session["json_data"] = result["json_data"]

                # Store information about the number of abstracts
                doc_manager.store_metadata(
                    session_id, {"abstract_count": len(df), "file_name": file.filename}
                )

                # Run QA pair generation in background
                qa_generation_started = run_qa_pair_generation(result["json_data"], session_id)

                return jsonify(
                    {
                        "success": True,
                        "session_id": session_id,
                        "count": len(df),
                        "multiple": True,
                        "message": f"Successfully processed {len(df)} abstracts",
                        "qa_generation_started": qa_generation_started
                    }
                )
            else:
                return jsonify({"success": False, "error": result["error"]})

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    except Exception as e:
        print(f"Error in submit_csv: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/submit-abstract", methods=["POST"])
def submit_abstract():
    try:
        data = request.get_json()
        abstract = data.get("text", "")

        if not abstract:
            return jsonify({"success": False, "error": "No abstract provided"})

        # Clean up old sessions
        doc_manager.clean_old_sessions()

        result = process_abstract_with_gpt(abstract)

        if result["success"]:
            session_id = str(uuid.uuid4())

            # Process for RAG
            rag_success = process_abstract_for_rag(abstract, session_id)
            if not rag_success:
                return jsonify(
                    {"success": False, "error": "Failed to process abstract for RAG"}
                )

            # Store in session
            session["current_session_id"] = session_id
            session["table_data"] = result["table_data"]
            session["json_data"] = result["json_data"]

            # Run QA pair generation in background
            qa_generation_started = run_qa_pair_generation(result["json_data"], session_id)

            return jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "message": "Abstract processed successfully",
                    "qa_generation_started": qa_generation_started
                }
            )
        else:
            return jsonify({"success": False, "error": result["error"]})

    except Exception as e:
        print(f"Error in submit_abstract: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get-table/<session_id>", methods=["GET"])
def get_table_data(session_id):
    """Get the table data for a specific session"""
    try:
        # Get table data from session
        table_data = session.get("table_data")

        if table_data:
            return jsonify({"success": True, "table_data": table_data})
        else:
            return jsonify(
                {"success": False, "error": "No data found for this session"}
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/download-json/<session_id>", methods=["GET"])
def download_json(session_id):
    """Download the JSON data for a specific session"""
    try:
        # Get JSON data from session
        json_data = session.get("json_data")

        if json_data:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                temp_file.write(json.dumps(json_data, indent=2).encode("utf-8"))

            # Prepare response
            response = flask.send_file(
                temp_file.name,
                as_attachment=True,
                download_name=f"hea_data_{session_id}.json",
                mimetype="application/json",
            )

            # Delete the file after sending
            response.call_on_close(lambda: os.unlink(temp_file.name))

            return response
        else:
            return jsonify(
                {"success": False, "error": "No JSON data found for this session"}
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/qa-pairs-status/<session_id>", methods=["GET"])
def qa_pairs_status(session_id):
    """Check the status of QA pair generation"""
    status = check_qa_pair_status(session_id)
    return jsonify(status)

@app.route("/api/get-qa-pairs/<session_id>", methods=["GET"])
def get_qa_pairs(session_id):
    """Get the QA pairs for a specific session"""
    try:
        # Check status
        status = check_qa_pair_status(session_id)
        
        if status["status"] != "completed":
            return jsonify({
                "success": False, 
                "error": f"QA pairs not ready. Current status: {status['status']}"
            })
        
        # Load the QA pairs
        result = load_qa_pairs(status["output_file"])
        
        if result["success"]:
            return jsonify({"success": True, "qa_pairs": result["table_data"]})
        else:
            return jsonify({"success": False, "error": result["error"]})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/download-qa-json/<session_id>", methods=["GET"])
def download_qa_json(session_id):
    """Download the QA pairs JSON file for a specific session"""
    try:
        # Check if the QA generation has completed
        status = check_qa_pair_status(session_id)
        
        if status["status"] != "completed":
            return jsonify({
                "success": False,
                "error": f"QA pairs not ready. Current status: {status['status']}"
            })
        
        # Get the path to the final QA pairs JSON file
        qa_process = session.get("qa_pair_process", {})
        output_dir = qa_process.get("output_dir")
        final_output_file = os.path.join(output_dir, "final_qa_pairs.json")
        
        if not os.path.exists(final_output_file):
            return jsonify({
                "success": False, 
                "error": "QA pairs file not found"
            })
            
        # Prepare response
        response = flask.send_file(
            final_output_file,
            as_attachment=True,
            download_name=f"qa_pairs_{session_id}.json",
            mimetype="application/json"
        )
        
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})



@app.route("/api/clean-cache", methods=["POST"])
def clean_cache_route():
    """API endpoint to clean model cache"""
    try:
        clear_model_cache()
        return jsonify({"success": True, "message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/toggle-model", methods=["POST"])
def toggle_model():
    """Toggle between local model and API model"""
    global use_api_model
    use_api_model = not use_api_model

    # Store the setting in the session
    session["use_api_model"] = use_api_model
    print(f"Model toggled, now using API: {use_api_model}")

    model_name = "API" if use_api_model else "Local (Phi-2)"
    return jsonify(
        {"success": True, "using_api": use_api_model, "model_name": model_name}
    )


@app.route("/api/get-model-status", methods=["GET"])
def get_model_status():
    """Get the current model status"""
    global use_api_model

    # Make sure session has the correct value
    if "use_api_model" not in session:
        session["use_api_model"] = DEFAULT_TO_API
        use_api_model = DEFAULT_TO_API
    else:
        use_api_model = session.get("use_api_model", DEFAULT_TO_API)

    model_name = "API" if use_api_model else "Local (Phi-2)"
    return jsonify({"using_api": use_api_model, "model_name": model_name})


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"answer": "Please enter a valid question."})

    try:
        # Get current session ID
        session_id = session.get("current_session_id")

        if not session_id:
            return jsonify(
                {
                    "answer": "Session expired or not found. Please submit an abstract first."
                }
            )

        # Check if document store has content for this session
        session_data = doc_manager.get_document(session_id)
        if not session_data or not session_data.get("chunks"):
            return jsonify(
                {"answer": "Please submit an abstract first before asking questions."}
            )

        # Check if we should use API from session, but default to API if not set
        global use_api_model
        if "use_api_model" in session:
            use_api_model = session.get("use_api_model", DEFAULT_TO_API)
        else:
            use_api_model = DEFAULT_TO_API
            session["use_api_model"] = DEFAULT_TO_API

        # Generate answer using RAG
        print(f"Processing question for session {session_id}: {question}")
        answer = generate_answer_with_retrieval(question, session_id)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify(
            {
                "answer": "I apologize, but I encountered an error. Please try asking your question differently."
            }
        )


if __name__ == "__main__":
    # Make sure the templates directory exists
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

    # Load the model when the app starts
    success = load_model()
    if not success:
        print(
            "WARNING: Model loading failed. The application may not function correctly."
        )

    # Initialize the document manager after embedder is loaded
    doc_manager = DocumentManager()

    app.run(debug=False, host="0.0.0.0", port=5000)
