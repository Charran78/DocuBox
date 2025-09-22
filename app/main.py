# app/main.py
from unittest import result
import streamlit as st
import os
import tempfile
import sys
import pathlib
import requests
import json
from datetime import datetime
import time
import base64
from PIL import Image
import random
import webbrowser
import sqlite3
import re
import math
import psutil
import torch
import GPUtil
from threading import Lock

resource_lock = Lock()

# Añadir el directorio actual al path para importaciones locales
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

# Importar módulos locales
try:
    from utils import process_files, extract_cover, scan_folder, process_folder_files, generate_file_preview, extract_metadata
    from chroma_manager import get_chroma_manager
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    # Fallback: define empty functions to avoid further errors
    def process_files(*args, **kwargs):
        return []
    def process_folder_files(*args, **kwargs):
        return []
    def extract_cover(*args, **kwargs):
        return False
    def scan_folder(*args, **kwargs):
        return []
    def generate_file_preview(*args, **kwargs):
        return ""
    def extract_metadata(*args, **kwargs):
        return {}
    class ChromaManager:
        def __init__(self, *args, **kwargs):
            pass
        def list_collections(self):
            return []
        def add_documents(self, *args, **kwargs):
            return 0
        def search(self, *args, **kwargs):
            return None
        def delete_collection(self, *args, **kwargs):
            return False
        def get_collection_documents(self, *args, **kwargs):
            return []

import ollama
from streamlit_option_menu import option_menu

# Configuración de la página
st.set_page_config(
    page_title="DocBox RAG - Local AI Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de Ollama
# OLLAMA_MODEL_PATH = "H:\\users\\xpite\\Desktop\\DocuBox\\ollama\\models"
OLLAMA_MODEL_PATH = "ollama/models"
ICONS_PATH = "assets/icons"  # Ruta para los iconos de tipos de archivo

# Estilos CSS para interfaz TV Box
st.markdown("""
<style>
    /* Estilos generales para interfaz TV Box */
    .tv-box {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    
    .tv-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }
    
    .tv-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .collection-card {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .preview-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .document-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    
    .document-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #4fc3f7, #29b6f6) !important;
        color: white !important;
        border: 2px solid #0288d1 !important;
        font-weight: bold !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(45deg, #78909c, #546e7a) !important;
        color: white !important;
        border: 1px solid #455a64 !important;
        opacity: 0.8;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }
    
    .action-button {
        background: linear-gradient(45deg, #4a69bd, #6a89cc) !important;
        margin: 5px;
        font-size: 0.8rem !important;
        padding: 8px 12px !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50, #4a69bd);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: rgba(74, 105, 189, 0.3);
        margin-left: 20%;
    }
    
    .chat-message.assistant {
        background-color: rgba(255, 107, 107, 0.3);
        margin-right: 20%;
    }
    
    /* Estilo para las pestañas de navegación */
    [data-testid="stHorizontalBlock"] {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 10px;
    }
    
    .thumbnail {
        border-radius: 8px;
        object-fit: cover;
        width: 100px;
        height: 140px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Estilo para el radio button de acciones */
    div[data-testid="stHorizontalBlock"] label {
        display: flex;
        justify-content: center;
        width: 100%;
        padding: 10px;
        border-radius: 8px;
        margin: 5px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stHorizontalBlock"] label:has(input:checked) {
        background: linear-gradient(45deg, #ff8e53, #ff6b6b);
        color: white !important;
        transform: scale(1.05);
        border-color: #ff6b6b;
    }
    
    div[data-testid="stHorizontalBlock"] label:has(input:not(:checked)) {
        background: #2c3e50;
        color: white;
        border: 1px solid #4a69bd;
    }
    
    /* Estilo para el botón seleccionado */
    .action-selected {
        background: linear-gradient(45deg, #ff8e53, #ff6b6b) !important;
        color: white !important;
        transform: scale(1.05);
        border-color: #ff6b6b !important;
    }

    /* Estilos base para recursos - se sobrescribirán dinámicamente */
    .resource-value {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #4fc3f7 !important;
    }
    
    .resource-label {
        font-size: 14px !important;
        color: #ffffff !important;
        opacity: 0.9;
    }
    
    /* Indicador de estado de GPU */
    .gpu-indicator {
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px 0;
        display: inline-block;
    }
    
    .gpu-active {
        background: linear-gradient(45deg, #00c853, #64dd17);
        color: white;
    }
    
    .gpu-inactive {
        background: linear-gradient(45deg, #f44336, #ff5252);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class LocalAIClient:
    def __init__(self, model_path):
        self.model_path = model_path
        self.available_models = self.get_available_models()
        self.response_cache = {}  # Cache para evitar repeticiones
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """Verifica si hay GPU disponible para aceleración"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU disponible: {gpu_count} dispositivos, {gpu_memory:.1f}GB")
                return True
            return False
        except:
            return False
    
    def get_available_models(self):
        """Obtiene modelos disponibles con múltiples métodos"""
        models = []
    
        # Método 1: Usar ollama.list()
        try:
            response = ollama.list()
            if hasattr(response, 'models'):
                models = [model.name for model in response.models]
            elif isinstance(response, dict) and 'models' in response:
                models = [model['name'] for model in response['models']]
        except:
            pass
    
        # Método 2: Usar API HTTP directa
        if not models:
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
            except:
                pass
    
        # Método 3: Modelos por defecto conocidos
        if not models:
            models = [
                'gemma2:2b-instruct-q4_K_M',
                # 'gemma2:2b-instruct', 
                # 'gemma2:2b',
                # 'gemma3:1b',
                # 'gemma2-2b:latest'
            ]
    
        return models
    
    def re_rank_chunks(self, query, chunks):
        """
        Re-ordena los chunks recuperados usando un LLM más pequeño para mejorar la relevancia.
        
        Args:
            query (str): La consulta del usuario.
            chunks (list): Lista de chunks de texto recuperados del vector store.
            model_name (str): El nombre del modelo a usar para re-ranking.
        
        Returns:
            list: Una lista de chunks re-ordenados de mayor a menor relevancia.
        """
        if not chunks:
            return []
            
        # model_to_use = 'gemma:2b' if 'gemma:2b' in self.available_models else 'gemma2:2b'
        model_to_use = next((m for m in self.available_models), None)
        if not model_to_use:
            return chunks # Retorna sin re-ranking si no hay modelo adecuado
            
        scoring_prompt = f"""
        INSTRUCCIONES CRÍTICAS: Eres un experto en encontrar información relevante.
        Recibirás una pregunta y un fragmento de texto.
        Tu única tarea es asignar una puntuación de 0 to 100 que indique la relevancia del fragmento para la pregunta.
        0 significa que no es relevante en absoluto, 100 significa que es extremadamente relevante.
        SOLO RESPONDE CON EL NÚMERO DE PUNTUACIÓN.
        
        PREGUNTA: {query}
        
        FRAGMENTO: {{chunk}}
        
        PUNTUACIÓN (0-100):
        """
        
        scored_chunks = []
        for chunk in chunks:
            try:
                response = ollama.chat(
                    model=model_to_use,
                    messages=[{'role': 'user', 'content': scoring_prompt.replace("{{chunk}}", chunk)}],
                    options={'num_predict': 10, 'temperature': 0.1}
                )
                score_str = response['message']['content'].strip()
                try:
                    score = int(re.search(r'\b\d{1,3}\b', score_str).group(0))
                except (ValueError, AttributeError):
                    score = 0 # Asigna 0 si el resultado no es un número válido
                scored_chunks.append({'text': chunk, 'score': score})
            except Exception as e:
                print(f"Error en el re-ranking: {e}")
                scored_chunks.append({'text': chunk, 'score': 0})
        
        # Ordenar los chunks por puntuación de mayor a menor
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return [item['text'] for item in scored_chunks]
        
    def generate_response(self, prompt, context="", specialized_action="default", max_tokens=1000, model_name=None, use_online=False):
        """Genera respuesta usando IA local con tipo de respuesta específico"""

        if use_online:
            return self._generate_online_response(prompt, context)
        
        # Parámetros optimizados para GPU
        gpu_params = {
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.8,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "num_ctx": 4096,  # Contexto aumentado para GPU
            "num_gpu": 50 if self.gpu_available else 0,  # Capas GPU para aceleración
        }
        
        # Crear clave única para la consulta
        cache_key = f"{prompt[:50]}_{hash(context) if context else 0}_{specialized_action}_{model_name}"
        
        # Verificar si ya está en cache
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        try:
            # Determinar qué modelo usar - priorizar modelos optimizados para GPU
            if model_name and model_name in self.available_models:
                model_to_use = model_name
            elif 'gemma2:2b-instruct-q4_K_M' in self.available_models:
                model_to_use = 'gemma2:2b-instruct-q4_K_M'  # Modelo cuantizado para mejor rendimiento
            # elif 'gemma3:1b' in self.available_models:
                # model_to_use = 'gemma3:1b'  # Modelo más pequeño y rápido
            elif self.available_models:
                model_to_use = self.available_models[0]
            else:
                return "Error: No hay modelos disponibles. Verifica que Ollama esté ejecutándose y que tengas modelos descargados."

            # Prompts profesionales y especializados
            prompts = {
                "búsqueda rápida": {
                    "instruction": "Responde de manera clara, concisa y directa. Sé un experto en la materia. Proporciona una respuesta completa basada en el contexto, pero sin divagar.",
                    "word_count": "Máximo 300 tokens. No repitas la pregunta."
                },
                "generar informe": {
                    "instruction": "Actúa como un redactor de informes profesional. Estructura un informe detallado con introducción, desarrollo de los puntos clave, y conclusiones. Extrae toda la información relevante del contexto y organízala de manera lógica.",
                    "word_count": "La respuesta debe ser un informe de entre 1500 y 2500 tokens. Desarrolla cada punto con detalle."
                },
                "crear tabla": {
                    "instruction": "Extrae la información relevante del contexto y organízala estrictamente en una tabla. Las columnas deben estar claramente definidas. Incluye todos los datos relevantes.",
                    "word_count": "No repitas la pregunta. Proporciona una tabla completa con toda la información disponible."
                },
                "análisis profundo": {
                    "instruction": "Actúa como un analista experto. Realiza un análisis exhaustivo del contexto, examinando causas, efectos y relaciones. Sintetiza las ideas principales en un resumen coherente. Profundiza en cada aspecto relevante.",
                    "word_count": "Proporciona una respuesta detallada de entre 1000 and 1500 tokens. Analiza minuciosamente la información."
                },
                "resumen": {
                    "instruction": "Genera un resumen ejecutivo del contexto proporcionado. Identifica los puntos clave y las conclusions principales. Sé conciso pero preciso. Incluye todos los aspectos importantes.",
                    "word_count": "La respuesta debe tener entre 1000 and 2000 tokens. Resume de manera completa pero concisa."
                },
                "respuesta detallada": {
                    "instruction": "Proporciona una respuesta detallada y exhaustiva, explorando todos los aspectos relevantes del contexto. Ofrece ejemplos y explicaciones. Desarrolla cada punto con profundidad.",
                    "word_count": "La respuesta debe tener entre 1500 and 1800 tokens. Extiéndete en cada aspecto relevante."
                }
            }
            
            # Construir prompt
            selected_prompt = prompts.get(specialized_action, prompts)
            
            full_prompt = f"""
            INSTRUCCIONES CRÍTICAS:
            - **ROL**: {selected_prompt['instruction']}
            - **REGLAS**:
                - UTILIZA TODA LA INFORMACIÓN RELEVANTE del contexto proporcionado.
                - DESARROLLA LA RESPUESTA de manera completa y exhaustiva.
                - SI LA RESPUESTA NO SE ENCUENTRA en el contexto, responde solo con: "No tengo información sobre esto en mis documentos."
                - NO inventes información, NO uses conocimiento externo.
                - ORGANIZA la información de manera lógica y coherente.
                - {selected_prompt['word_count']}

            ---
            
            CONTEXTO (documentos del usuario):
            {context}
            
            ---
            
            PREGUNTA: {prompt}
            
            ---
            
            RESPUESTA:
            """
            
            # Generar la respuesta con Ollama - optimizado para GPU
            response = ollama.chat(
                model=model_to_use,
                messages=[{'role': 'user', 'content': full_prompt}],
                options={
                    'num_predict': gpu_params["max_tokens"],
                    'temperature': gpu_params["temperature"],
                    'top_p': gpu_params["top_p"],
                    'repeat_penalty': gpu_params["repeat_penalty"],
                    'top_k': gpu_params["top_k"],
                    'num_ctx': gpu_params["num_ctx"],
                    'num_gpu': gpu_params["num_gpu"],
                    'stop': ["PREGUNTA:", "CONTEXTO:", "###"]
                }
            )
            
            result = response['message']['content']
            
            # 🔥 PROTECCIÓN ANTI-REPETICIÓN Y LIMPIEZA FINAL
            result = self._remove_repetitions(result)
            
            # Guardar en cache
            self.response_cache[cache_key] = result
            return result
            
        except Exception as e:
            return f"Error: {str(e)}. Verifica que Ollama esté ejecutándose."

    def _remove_repetitions(self, text):
        """Elimina repeticiones del texto generado"""
        if not text:
            return text
        
        # Dividir en oraciones
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return text
        
        # Eliminar repeticiones consecutivas
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            # Normalizar y verificar similitud
            normalized = ' '.join(sentence.lower().split())
            if normalized and normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
            
            # Limitar a 40 oraciones máximo
            if len(unique_sentences) >= 40:
                break
        
        result = ' '.join(unique_sentences).strip()
        if result and not result.endswith('.'):
            result += '.'
        
        # Limitar longitud total
        if len(result) > 2500:
            result = result[:2500] + "..."
        
        return result

    def _generate_online_response(self, prompt, context):
        """Genera respuesta con IA online abriendo búsqueda en Google"""
        try:
            # Construir la URL de búsqueda de Google
            search_query = f"{prompt} {context}".strip() if context else prompt
            search_query = search_query.replace(" ", "+")
            google_url = f"https://www.google.com/search?q={search_query}"
            
            # Abrir en el navegador predeterminado
            webbrowser.open(google_url)
            
            return f"🔍 Búsqueda online iniciada: '{prompt}'. " \
                   f"Se ha abierto tu navegador predeterminado con los resultados de Google."
            
        except Exception as e:
            return f"❌ Error al realizar la búsqueda online: {str(e)}"

def get_file_icon(file_path):
    """Devuelve un icono basado en el tipo de archivo"""
    extension = os.path.splitext(file_path)[1].lower()
    
    icon_map = {
        '.pdf': '📕',
        '.doc': '📘',
        '.docx': '📘',
        '.txt': '📄',
        '.xls': '📊',
        '.xlsx': '📊',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.png': '🖼️',
        '.epub': '📖',
    }
    
    return icon_map.get(extension, '📁')

def get_file_icon_path(file_path):
    """Devuelve la ruta al icono PNG basado en el tipo de archivo"""
    extension = os.path.splitext(file_path)[1].lower().replace('.', '')
    icon_path = os.path.join(ICONS_PATH, f"{extension}.png")
    
    # Si no existe el icono específico, usar el icono por defecto
    if not os.path.exists(icon_path):
        icon_path = os.path.join(ICONS_PATH, "default.png")
    
    return icon_path

def display_collection_cards(collections, chroma_manager):
    """Muestra las colecciones como tarjetas estilo TV Box"""
    if not collections:
        st.info("No hay colecciones creadas. Ve a 'Ingesta de Datos' para crear una.")
        return
    
    cols = st.columns(3)
    for i, collection in enumerate(collections):
        with cols[i % 3]:
            # Crear tarjeta de colección
            with st.container():
                st.markdown(f"""
                <div class="tv-card">
                    <h3>{get_file_icon('')} {collection['name']}</h3>                                    
                    <p>📊 {collection.get('count', 0)} documentos</p>
                    <p>🧩 {collection.get('chunk_count', 0)} chunks</p>
                    <p>🔄 {collection.get('updated_at', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Botones de acción
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Explorar", key=f"explore_{collection['name']}"):
                        st.session_state.selected_collection = collection['name']
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{collection['name']}"):
                        if chroma_manager.delete_collection(collection['name']):
                            st.success(f"Colección '{collection['name']}' eliminada")
                            time.sleep(1)
                            st.rerun()

def display_document_preview(file_path, max_preview_length=500):
    """Muestra una vista previa del documento"""
    try:
        preview = generate_file_preview(file_path)
        if preview:
            # Acortar preview si es necesario
            if len(preview) > max_preview_length:
                preview = preview[:max_preview_length] + "..."
            
            st.markdown(f"""
            <div class="preview-container">
                <h4>Vista Previa:</h4>
                <p>{preview}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generando vista previa: {e}")

def display_collection_documents(collection_name, chroma_manager):
    """Muestra todos los documentos de una colección con sus carátulas/iconos"""
    try:
        documents = chroma_manager.get_collection_documents(collection_name)
        
        if not documents:
            st.info("No se encontraron documentos en esta colección.")
            return
        
        st.markdown(f"### 📄 Documentos en '{collection_name}'")
        
        # Filtro por tipo de documento
        doc_types = list(set([doc.get('type', 'unknown') for doc in documents]))
        doc_types.sort()
        selected_type = st.selectbox("Filtrar por tipo:", ["Todos"] + doc_types)
        
        # Búsqueda por nombre
        search_query = st.text_input("Buscar documento por nombre:", "")
        
        # Paginación
        items_per_page = 6
        total_pages = math.ceil(len(documents) / items_per_page) if documents else 1
        page = st.number_input("Página", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(documents))
        
        # Filtrar documentos
        filtered_docs = documents
        if selected_type != "Todos":
            filtered_docs = [doc for doc in filtered_docs if doc.get('type', 'unknown') == selected_type]
        if search_query:
            filtered_docs = [doc for doc in filtered_docs if search_query.lower() in doc.get('name', '').lower()]
        
        # Mostrar documentos
        if not filtered_docs:
            st.info("No hay documentos que coincidan con los filtros.")
            return
        
        # Mostrar documentos paginados
        for i in range(start_idx, min(end_idx, len(filtered_docs))):
            doc = filtered_docs[i]
            file_path = doc.get('file_path', '')
            doc_name = doc.get('name', 'Sin nombre')
            doc_type = doc.get('type', 'unknown')
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Mostrar carátula o icono
                try:
                    if doc_type in ['pdf', 'epub']:
                        cover_path = extract_cover(file_path)
                        if cover_path and os.path.exists(cover_path):
                            st.image(cover_path, use_container_width=True, caption=doc_name)
                        else:
                            icon_path = get_file_icon_path(file_path)
                            st.image(icon_path, use_container_width=True, caption=doc_name)
                    else:
                        icon_path = get_file_icon_path(file_path)
                        st.image(icon_path, use_container_width=True, caption=doc_name)
                except Exception as e:
                    st.error(f"Error cargando imagen: {e}")
            
            with col2:
                st.markdown(f"""
                <div class="document-card">
                    <h4>{get_file_icon(file_path)} {doc_name}</h4>
                    <p><strong>Tipo:</strong> {doc_type}</p>
                    <p><strong>Tamaño:</strong> {doc.get('size', 'N/A')}</p>
                    <p><strong>Fecha:</strong> {doc.get('date', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Botones de acción para el documento
                col21, col22, col23 = st.columns(3)
                with col21:
                    if st.button("👀 Abrir", key=f"open_{i}_{collection_name}"):
                        try:
                            if os.path.exists(file_path):
                                webbrowser.open(f"file://{file_path}")
                            else:
                                st.error("El archivo no existe en la ruta especificada.")
                        except Exception as e:
                            st.error(f"No se pudo abrir el archivo: {e}")
                
                with col22:
                    if st.button("📋 Vista previa", key=f"preview_{i}_{collection_name}"):
                        st.session_state.preview_doc = file_path
                
                with col23:
                    if st.button("🗑️ Eliminar", key=f"remove_{i}_{collection_name}"):
                        if chroma_manager.remove_document(collection_name, file_path):
                            st.success("Documento eliminado de la colección")
                            time.sleep(1)
                            st.rerun()
            
            # Mostrar vista previa si se solicita
            if st.session_state.get('preview_doc') == file_path:
                with st.expander("Vista previa del documento", expanded=True):
                    display_document_preview(file_path)
        
        # Controles de paginación
        if total_pages > 1:
            st.markdown(f"**Página {page} de {total_pages} - {len(filtered_docs)} documentos**")
            prev_col, next_col, _ = st.columns([1, 1, 3])
            with prev_col:
                if page > 1 and st.button("◀ Anterior"):
                    st.session_state.current_page = page - 1
                    st.rerun()
            with next_col:
                if page < total_pages and st.button("Siguiente ▶"):
                    st.session_state.current_page = page + 1
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error mostrando documentos: {e}")

def display_structured_data():
    """Muestra la información estructurada de la base de datos SQL"""
    st.markdown("### 📊 Documentos Estructurados")
    
    try:
        conn = sqlite3.connect("data/documentos.db")
        cursor = conn.cursor()
        
        # Obtener estadísticas
        cursor.execute("SELECT COUNT(*), doc_type FROM documentos GROUP BY doc_type")
        stats = cursor.fetchall()
        
        if not stats:
            st.info("No hay documentos estructurados en la base de datos.")
            return
        
        # Mostrar estadísticas
        col1, col2, col3 = st.columns(3)
        total_docs = sum([count for count, _ in stats])
        
        with col1:
            st.metric("Total Documentos", total_docs)
        with col2:
            st.metric("Tipos Diferentes", len(stats))
        with col3:
            most_common = max(stats, key=lambda x: x[0])[1] if stats else "N/A"
            st.metric("Tipo Más Común", most_common)
        
        # Selector de tipo de documento
        doc_types = [row[1] for row in stats]
        selected_type = st.selectbox("Seleccionar tipo de documento:", ["Todos"] + doc_types)
        
        # Consultar documentos
        if selected_type == "Todos":
            cursor.execute("""
                SELECT d.file_name, d.doc_type, d.language, d.file_size, 
                       f.numero_factura, f.fecha_emision, f.total,
                       l.titulo, l.autor, l.editorial
                FROM documentos d
                LEFT JOIN facturas f ON d.id = f.documento_id
                LEFT JOIN libros l ON d.id = l.documento_id
                ORDER BY d.processed_at DESC
            """)
        else:
            cursor.execute("""
                SELECT d.file_name, d.doc_type, d.language, d.file_size, 
                       f.numero_factura, f.fecha_emision, f.total,
                       l.titulo, l.autor, l.editorial
                FROM documentos d
                LEFT JOIN facturas f ON d.id = f.documento_id
                LEFT JOIN libros l ON d.id = l.documento_id
                WHERE d.doc_type = ?
                ORDER BY d.processed_at DESC
            """, (selected_type,))
        
        documents = cursor.fetchall()
        
        # Mostrar documentos en una tabla
        if documents:
            st.markdown(f"#### 📋 {len(documents)} documentos encontrados")
            
            for i, doc in enumerate(documents):
                with st.expander(f"{doc[0]} ({doc[1]})", expanded=i < 3):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Información Básica:**")
                        st.write(f"Nombre: {doc[0]}")
                        st.write(f"Tipo: {doc[1]}")
                        st.write(f"Idioma: {doc[2]}")
                        st.write(f"Tamaño: {doc[3]} bytes")
                    
                    with col2:
                        if doc[1] == "factura" and doc[4]:
                            st.write("**Información de Factura:**")
                            st.write(f"Número: {doc[4]}")
                            st.write(f"Fecha: {doc[5]}")
                            st.write(f"Total: {doc[6]}")
                        elif doc[1] == "libro" and doc[7]:
                            st.write("**Información de Libro:**")
                            st.write(f"Título: {doc[7]}")
                            st.write(f"Autor: {doc[8]}")
                            st.write(f"Editorial: {doc[9]}")
                        else:
                            st.write("**Sin información estructurada adicional**")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error accediendo a la base de datos: {e}")

def get_gpu_info():
    """Obtiene información detallada sobre la GPU disponible"""
    gpu_info = {}
    
    try:
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['devices'] = []
            
            for i in range(gpu_info['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_gb': device_props.total_memory / (1024**3),
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                })
                
            # Memoria actualmente utilizada
            gpu_info['memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            gpu_info['memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
        else:
            gpu_info['available'] = False
            
    except Exception as e:
        gpu_info['error'] = str(e)
        
    return gpu_info

def main():
    # Inicializar session_state
    if 'scanned_files' not in st.session_state:
        st.session_state.scanned_files = []
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'preview_doc' not in st.session_state:
        st.session_state.preview_doc = None
    if 'specialized_action' not in st.session_state:
        st.session_state.specialized_action = "búsqueda rápida"
        
    # Inicializar managers
    chroma_manager = get_chroma_manager()
    ai_client = LocalAIClient(OLLAMA_MODEL_PATH)
    
    # Cargar modelos disponibles
    if not st.session_state.available_models:
        st.session_state.available_models = ai_client.available_models
    
    # Obtener información de GPU
    gpu_info = get_gpu_info()
    

    # Sidebar con navegación
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h1>📦 DocBox RAG</h1>
            <p>Tu asistente de IA local para documentos</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")        
            
        selected = option_menu(
            menu_title="Navegación",
            options=[" Inicio", " Ingesta de Datos", " Busqueda", " Chat con IA", " Documentos Estructurados", " Configuracion"],
            icons=["house", "folder", "search", "chat", "database", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "rgba(0,0,0,0.2)"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#ff6b6b"},
            }
        )

         # Mostrar estado de GPU
        if gpu_info.get('available', False):
            gpu_status = f"<span class='gpu-indicator gpu-active'>✅ GPU Disponible: ({gpu_info['devices'][0]['name']})"
            gpu_status += f"<br><strong>Memoria Total:</strong> {gpu_info['devices'][0]['total_memory_gb']:.2f} GB"
            gpu_status += f"<br><strong>Procesadores:</strong> {gpu_info['devices'][0]['multi_processor_count']}"
            gpu_status += f"<br><strong>Versión CUDA:</strong> {gpu_info['devices'][0]['major']}.{gpu_info['devices'][0]['minor']}</span>"
        
        else:
            gpu_status = "<span class='gpu-indicator gpu-inactive'>GPU ❌ (Solo CPU)</span>"

        # Mostrar información de GPU       
        st.markdown(f"🖥️ Información del Equipo: {gpu_status}", unsafe_allow_html=True)        
        
        collections = chroma_manager.list_collections()
        total_docs = sum(collection.get('count', 0) for collection in collections)
        total_chunks = sum(collection.get('chunk_count', 0) for collection in collections)
        
        st.markdown("---")
        st.markdown("### 📊 Estadísticas del Sistema")
        st.markdown(f"""
        - 🗂️ Colecciones: **{len(collections)}**
        - 📄 Documentos: **{total_docs}**
        - 🧩 Chunks: **{total_chunks}**
        - 🌟 Modelos: **{len(st.session_state.available_models)}**
        """)

           

        st.markdown("---")
    
    # Página de Inicio
    if selected == " Inicio":
        st.markdown("""
        <div class="tv-box">
            <h1 style="text-align: center;">🎯 DocBox RAG - Asistente de IA Local</h1>
            <p style="text-align: center;">Explora, busca y consulta tus documentos con privacidad total</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Si hay una colección seleccionada, mostrar sus documentos
        if st.session_state.selected_collection:
            if st.button("← Volver a la vista de colecciones"):
                st.session_state.selected_collection = None
                st.session_state.preview_doc = None
                st.rerun()
            
            display_collection_documents(st.session_state.selected_collection, chroma_manager)
        else:
            st.markdown("### 📦 Tus Colecciones")
            display_collection_cards(collections, chroma_manager)
            
            # Estadísticas rápidas
            if collections:
                st.markdown("### 📈 Estadísticas Rápidas")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Colecciones", len(collections))
                with col2:
                    st.metric("Documentos Totales", total_docs)
                with col3:
                    st.metric("Chunks de Texto", total_chunks)
            
            # Acciones rápidas
            st.markdown("### ⚡ Acciones Rápidas")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Actualizar Modelos", use_container_width=True):
                    st.session_state.available_models = ai_client.get_available_models()
                    st.success("Modelos actualizados")
                    st.rerun()
            
            with col2:
                if st.button("📊 Ver Estadísticas", use_container_width=True):
                    selected = "⚙️ Configuración"
                    st.rerun()
            
            with col3:
                if st.button("➕ Nueva Colección", use_container_width=True):
                    selected = "📁 Ingesta de Datos"
                    st.rerun()
    
    # Página de Ingesta de Datos
    elif selected == " Ingesta de Datos":
        st.markdown("""
        <div class="tv-box">
            <h2>📁 Ingesta de Documentos</h2>
            <p>Sube o escanea documentos para crear nuevas colecciones</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            collection_name = st.text_input("Nombre de la colección:", "mi_coleccion")
            
            # Sección de subida de archivos
            st.markdown("#### 📤 Subir Archivos")
            uploaded_files = st.file_uploader(
                "Selecciona documentos (PDF, Word, Texto, Excel, Imágenes)",
                type=['txt', 'pdf', 'docx', 'doc', 'xls', 'xlsx', 'jpg', 'jpeg', 'png', 'epub'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.success(f"📄 {len(uploaded_files)} archivos seleccionados")
                
                # Mostrar preview del primer archivo
                if uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_files[0].name)[1]) as tmp_file:
                        tmp_file.write(uploaded_files[0].getvalue())
                        display_document_preview(tmp_file.name)
                        os.unlink(tmp_file.name)
            
            if uploaded_files and collection_name:
                if st.button("🚀 Procesar Documentos Subidos", type="primary", use_container_width=True):
                    with st.spinner("Procesando documentos..."):
                        processed_files = process_files(uploaded_files, collection_name)
                        
                        if processed_files:
                            try:
                                count = chroma_manager.add_documents(collection_name, processed_files)
                                st.success(f"✅ {count} documentos procesados en '{collection_name}'")
                                time.sleep(2)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error al crear la colección: {str(e)}")
                        else:
                            st.error("❌ No se pudieron procesar los documentos.")
            
            # SECCIÓN DE ESCANEO DE CARPETAS
            st.markdown("#### 📂 Escaneo de Carpetas")
            folder_path = st.text_input("Ruta de la carpeta:", value="H:\\ruta\\a\\tu\\carpeta", placeholder="Ej: H:\\ruta\\a\\tu\\carpeta")
            
            if folder_path and st.button("🔍 Escanear Carpeta", use_container_width=True):
                with st.spinner("Escaneando carpeta..."):
                    files = scan_folder(folder_path)
                    if files:
                        st.session_state.scanned_files = files
                        st.success(f"✅ {len(files)} archivos encontrados")
                    else:
                        st.warning("No se encontraron archivos procesables")
            
            # Botón para procesar los archivos escaneados
            if st.session_state.scanned_files and st.button("🚀 Procesar Archivos Escaneados", use_container_width=True):
                with st.spinner("Procesando archivos de la carpeta..."):
                    processed_files = process_folder_files(st.session_state.scanned_files, collection_name)
                    
                    if processed_files:
                        try:
                            count = chroma_manager.add_documents(collection_name, processed_files)
                            st.success(f"✅ {count} documentos procesados desde carpeta en '{collection_name}'")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error al crear la colección: {str(e)}")
                    else:
                        st.error("❌ No se pudieron procesar los archivos de la carpeta.")
        
        with col2:
            st.markdown("#### 📦 Colecciones Existentes")
            if collections:
                for collection in collections:
                    with st.container():
                        st.markdown(f"""
                        <div class="collection-card">
                            <h4>{collection['name']}</h4>
                            <p>📊 {collection.get('total_docs', 0)} documentos</p>
                            <p>🧩 {collection.get('chunk_count', 0)} chunks</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No hay colecciones creadas")
    
    # Página de Búsqueda
    elif selected == " Busqueda":
        st.markdown("""
        <div class="tv-box">
            <h2>🔍 Búsqueda Semántica</h2>
            <p>Encuentra información en tus documentos usando búsqueda semántica</p>
        </div>
        """, unsafe_allow_html=True)
        
        if collections:
            # Selector de colección
            collection_names = [col['name'] for col in collections]
            if st.session_state.selected_collection and st.session_state.selected_collection in collection_names:
                default_index = collection_names.index(st.session_state.selected_collection)
            else:
                default_index = 0
                
            selected_collection = st.selectbox("Selecciona colección:", collection_names, index=default_index)
            
            # Búsqueda
            query = st.text_input("Consulta de búsqueda:", placeholder="Escribe tu pregunta...")
            n_results = st.slider("Número de resultados:", 1, 20, 10)
            
            if query and selected_collection and st.button("🔍 Buscar", use_container_width=True):
                with st.spinner("Buscando..."):
                    results = chroma_manager.search(selected_collection, query, n_results)
                    
                    if results and results['documents']:
                        st.success(f"📋 {len(results['documents'][0])} resultados encontrados")
                        
                        # Re-ranking de chunks
                        chunks = results['documents'][0]
                        ranked_chunks = ai_client.re_rank_chunks(query, chunks)
                        
                        for i, chunk in enumerate(ranked_chunks[:n_results]):
                            with st.expander(f"Resultado {i+1} (Relevancia: {i+1}/{len(ranked_chunks)})"):
                                st.text(chunk[:1000] + "..." if len(chunk) > 1000 else chunk)
                    else:
                        st.warning("No se encontraron resultados")
        else:
            st.info("Primero crea una colección en la pestaña de Ingesta")
    
    # Página de Chat con IA
    elif selected == " Chat con IA":
        st.markdown("""
        <div class="tv-box">
            <h2>💬 Chat con IA</h2>
            <p>Conversa con la IA usando el contexto de tus documentos</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Selector de modo IA
        ai_mode = st.radio(
            "Modo de IA:",
            ["🔒 Solo Local (Privado)", "🌎 Online + Local"],
            horizontal=True
        )
        
        use_online = ai_mode == "🌎 Online + Local"
        
        if use_online:
            st.warning("⚠️ El modo online enviará datos externamente. Usa con cuidado.")
        
        # Selección de modelo
        if st.session_state.available_models:
            selected_model = st.selectbox(
                "Selecciona el modelo:",
                st.session_state.available_models,
                index=0
            )
        else:
            selected_model = None
            st.warning("No se detectaron modelos. Ejecuta 'ollama pull gemma2:2b'")
        
        # Selección de colección para contexto
        context_collection = None
        
        if collections:
            collection_names = [col['name'] for col in collections]
            selected_collection = st.selectbox("Usar colección como contexto:", ["Ninguna"] + collection_names)
            
            if selected_collection != "Ninguna":
                context_collection = selected_collection
        
        # Selector de acciones especializadas
        st.markdown("#### 🛠️ Acciones Especializadas")
        
        # Definir acciones y sus iconos
        actions = {
            "búsqueda rápida": "⚡",
            "generar informe": "📊",
            "crear tabla": "📋",
            "análisis profundo": "🔍",
            "resumen": "📝",
            "respuesta detallada": "💬"
        }
        
        # Crear botones horizontales para las acciones
        cols = st.columns(len(actions))

        for i, (action, icon) in enumerate(actions.items()):
            with cols[i]:
                is_selected = st.session_state.specialized_action == action
        
                # Determinar el color del botón según si está seleccionado o no
                button_type = "primary" if is_selected else "secondary"
        
                if st.button(
                    f"{icon} {action.capitalize()}",
                    use_container_width=True,
                    type=button_type,
                    key=f"action_{action}"
                ):
                    st.session_state.specialized_action = action
                    st.rerun()
        
        # Historial de chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input de usuario
        if prompt := st.chat_input("Escribe tu pregunta..."):
            # Añadir mensaje de usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Buscar contexto relevante con re-ranking
            context = ""
            if context_collection and prompt:
                try:
                    # Buscar más resultados para re-ranking
                    search_results = chroma_manager.search(context_collection, prompt, n_results=20)
                    if search_results and search_results['documents']:
                        # Extraer chunks
                        chunks = search_results['documents'][0]
                        
                        # Aplicar re-ranking
                        ranked_chunks = ai_client.re_rank_chunks(prompt, chunks)
                        
                        # Tomar los 8 chunks más relevantes
                        best_chunks = ranked_chunks[:8]
                        
                        context_parts = []
                        for i, chunk in enumerate(best_chunks):
                            context_parts.append(f"--- DOCUMENTO {i+1} ---\n{chunk}\n")
                        
                        context = "\n".join(context_parts)
                        
                        with st.expander("📋 Ver contexto utilizado"):
                            st.text(context[:1500] + "..." if len(context) > 1500 else context)
                    else:
                        st.warning("⚠️ No se encontraron documentos para tu pregunta")
                        context = ""
                except Exception as e:
                    st.error(f"❌ Error buscando contexto: {e}")
                    context = ""
            
            # Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("⚡ Pensando..."):
                    response = ai_client.generate_response(
                        prompt, context, 
                        specialized_action=st.session_state.specialized_action,
                        model_name=selected_model, 
                        use_online=use_online
                    )
                    
                    # Mostrar la respuesta de forma progresiva
                    assistant_response_placeholder = st.empty()
                    full_response_chunks = []
        
                    # Dividir la respuesta en chunks para simular escritura
                    for chunk in response.split(" "):
                        full_response_chunks.append(chunk + " ")
                        assistant_response_placeholder.markdown("".join(full_response_chunks) + "▌")
                        time.sleep(0.05)
        
                    # Mostrar la respuesta completa
                    assistant_response_placeholder.markdown("".join(full_response_chunks))
    
                    # Añadir respuesta al historial
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Botón para limpiar chat
        if st.button("🧹 Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Página de Documentos Estructurados
    elif selected == " Documentos Estructurados":
        display_structured_data()
    
    # Página de Configuración
    elif selected == " Configuracion":
        st.markdown("""
        <div class="tv-box">
            <h2>⚙️ Configuración del Sistema</h2>
            <p>Gestiona tu configuración y preferencias</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Colecciones", "🤖 Modelos", "⚡ Sistema", "🎮 GPU"])
        
        with tab1:
            st.markdown("#### 📊 Gestión de Colecciones")
            if collections:
                for collection in collections:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="collection-card">
                            <h4>{collection['name']}</h4>
                            <p>📊 {collection.get('count', 0)} documentos | 🧩 {collection.get('chunk_count', 0)} chunks</p>
                            <p>🔄 {collection.get('updated_at', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("📊 Stats", key=f"stats_{collection['name']}"):
                            stats = chroma_manager.get_collection_stats(collection['name'])
                            st.info(f"Estadísticas de {collection['name']}: {stats}")
                    with col3:
                        if st.button("🗑️", key=f"delete_{collection['name']}"):
                            if chroma_manager.delete_collection(collection['name']):
                                st.success("Colección eliminada")
                                time.sleep(1)
                                st.rerun()
            else:
                st.info("No hay colecciones creadas")
        
        with tab2:
            st.markdown("#### 🤖 Modelos de IA Disponibles")
            if st.session_state.available_models:
                for model in st.session_state.available_models:
                    st.write(f"• {model}")
            else:
                st.warning("No se detectaron modelos de IA")
            
            if st.button("🔄 Actualizar Modelos Disponibles"):
                st.session_state.available_models = ai_client.get_available_models()
                st.rerun()
        
        with tab3:
            st.markdown("#### ⚡ Información del Sistema")
            st.write(f"**Ruta de modelos:** {OLLAMA_MODEL_PATH}")
            st.write(f"**Directorio de datos:** {os.path.abspath('data/chromadb')}")
            st.write(f"**Ruta de iconos:** {ICONS_PATH}")

            st.markdown("#### 🗑️ Limpieza del Sistema")
            if st.button("🧹 Limpiar Caché de Chat"):
                st.session_state.messages = []
                st.success("Caché de chat limpiado")

            if st.button("🗑️ Limpiar Archivos Temporales"):
                st.success("Archivos temporales limpiados")
                
        with tab4:
            st.markdown("#### 🎮 Información de GPU")
            
            if gpu_info.get('available', False):
                st.success("✅ GPU detectada y disponible para aceleración")
                
                for i, device in enumerate(gpu_info['devices']):
                    st.markdown(f"**Dispositivo {i+1}:** {device['name']}")
                    st.markdown(f"- **Memoria total:** {device['total_memory_gb']:.2f} GB")
                    st.markdown(f"- **Arquitectura:** {device['major']}.{device['minor']}")
                    st.markdown(f"- **MP Count:** {device['multi_processor_count']}")
                
                st.markdown(f"**Memoria en uso:** {gpu_info.get('memory_allocated', 0):.2f} GB")
                st.markdown(f"**Memoria reservada:** {gpu_info.get('memory_reserved', 0):.2f} GB")
                
                # Recomendaciones de optimización
                st.markdown("#### ⚡ Recomendaciones de Optimización")
                if gpu_info['devices'][0]['total_memory_gb'] < 4:
                    st.warning("Tu GPU tiene poca memoria. Considera usar modelos más pequeños como 'gemma2:2b' o 'gemma3:1b'")
                else:
                    st.success("Tu GPU tiene suficiente memoria para modelos medianos como 'gemma2:2b'")
                    
            else:
                st.error("❌ No se detectó GPU disponible. La inferencia se realizará en CPU, lo que será más lento.")
                if 'error' in gpu_info:
                    st.error(f"Error: {gpu_info['error']}")

if __name__ == "__main__":
    main()