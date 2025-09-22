# utils.py
from bs4 import BeautifulSoup
import os
import magic
import PyPDF2
import docx2txt
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
import tempfile
from langdetect import detect, LangDetectException
import shutil
import mimetypes
from typing import List, Dict, Tuple
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime
import psutil
import time
import ebooklib
from ebooklib import epub
import xlrd
import csv
import sqlite3
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Configuración de rutas
ICONS_PATH = "assets/icons"
os.makedirs(ICONS_PATH, exist_ok=True)

# Modelo de clasificación de documentos
CLASSIFIER_PATH = "models/document_classifier.pkl"
os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)

def init_db(db_path="data/documentos.db"):
    """Inicializa la base de datos SQLite para documentos estructurados"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tabla principal de documentos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_name TEXT,
            file_type TEXT,
            file_size INTEGER,
            doc_type TEXT,
            language TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    ''')
    
    # Tabla para facturas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS facturas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id INTEGER,
            numero_factura TEXT,
            fecha_emision TEXT,
            emisor TEXT,
            receptor TEXT,
            total REAL,
            moneda TEXT,
            FOREIGN KEY (documento_id) REFERENCES documentos (id)
        )
    ''')
    
    # Tabla para libros
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS libros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id INTEGER,
            titulo TEXT,
            autor TEXT,
            año_publicacion INTEGER,
            editorial TEXT,
            isbn TEXT,
            FOREIGN KEY (documento_id) REFERENCES documentos (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_document_info(db_path, file_path, file_name, file_type, file_size, doc_type, language, metadata):
    """Guarda la información básica del documento en la base de datos"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    metadata_str = json.dumps(metadata, ensure_ascii=False)
    
    cursor.execute('''
        INSERT OR REPLACE INTO documentos 
        (file_path, file_name, file_type, file_size, doc_type, language, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (file_path, file_name, file_type, file_size, doc_type, language, metadata_str))
    
    conn.commit()
    conn.close()

def save_structured_data(db_path, file_path, structured_data, doc_type):
    """Guarda información estructurada específica según el tipo de documento"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Obtener el ID del documento
    cursor.execute('SELECT id FROM documentos WHERE file_path = ?', (file_path,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return False
    
    doc_id = result[0]
    
    if doc_type == "factura":
        cursor.execute('''
            INSERT OR REPLACE INTO facturas 
            (documento_id, numero_factura, fecha_emision, emisor, receptor, total, moneda)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            structured_data.get('numero_factura'),
            structured_data.get('fecha_emision'),
            structured_data.get('emisor'),
            structured_data.get('receptor'),
            structured_data.get('total'),
            structured_data.get('moneda')
        ))
    
    elif doc_type == "libro":
        cursor.execute('''
            INSERT OR REPLACE INTO libros 
            (documento_id, titulo, autor, año_publicacion, editorial, isbn)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            structured_data.get('titulo'),
            structured_data.get('autor'),
            structured_data.get('año_publicacion'),
            structured_data.get('editorial'),
            structured_data.get('isbn')
        ))
    
    conn.commit()
    conn.close()
    return True

def train_document_classifier():
    """Entrena un clasificador de documentos con Scikit-learn"""
    # Datos de ejemplo para entrenamiento (deberías expandir esto)
    training_data = [
        ("factura", "factura numero fecha emisor receptor total impuestos"),
        ("factura", "invoice number date seller buyer amount taxes"),
        ("libro", "libro título autor editorial año publicación isbn"),
        ("libro", "book title author publisher year publication isbn"),
        ("contrato", "contrato partes cláusulas fecha firma obligaciones"),
        ("articulo", "artículo abstract introducción método resultados discusión"),
        ("manual", "manual instrucciones uso procedimiento pasos seguridad"),
        ("formulario", "formulario campos nombre apellido email teléfono"),
        ("presentacion", "presentación diapositivas título contenido imágenes"),
        ("hoja_calculo", "excel tabla filas columnas datos fórmulas"),
    ]
    
    texts = [text for _, text in training_data]
    labels = [label for label, _ in training_data]
    
    # Vectorizar textos
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Entrenar clasificador
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    
    # Guardar modelo y vectorizer
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier}, CLASSIFIER_PATH)
    
    return vectorizer, classifier

def load_document_classifier():
    """Carga el clasificador de documentos entrenado"""
    if not os.path.exists(CLASSIFIER_PATH):
        return train_document_classifier()
    
    model_data = joblib.load(CLASSIFIER_PATH)
    return model_data['vectorizer'], model_data['classifier']

def classify_document_scikit(text):
    """Clasifica el documento usando Scikit-learn"""
    try:
        vectorizer, classifier = load_document_classifier()
        
        # Preprocesar texto
        cleaned_text = ' '.join(re.findall(r'\b\w+\b', text.lower())[:100])
        
        # Vectorizar y predecir
        X = vectorizer.transform([cleaned_text])
        prediction = classifier.predict(X)[0]
        
        return prediction
    except Exception as e:
        print(f"Error en clasificación Scikit-learn: {e}")
        return "otro"

def classify_document_llm(text, model_name="gemma2:2b-instruct-q4_K_M"):
    """Clasifica el documento usando LLM como respaldo"""
    prompt = f"""
    Clasifica el siguiente documento en una de estas categorías: 
    - factura
    - libro
    - contrato
    - articulo
    - manual
    - formulario
    - presentacion
    - hoja_calculo
    - imagen
    - otro

    Responde SOLO con el nombre de la categoría en minúsculas y sin puntuación.

    Fragmento del documento:
    {text[:500]}
    """
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 10}
        )
        category = response['message']['content'].strip().lower()
        
        # Limpiar y validar la respuesta
        category = re.sub(r'[^a-z_]', '', category)
        valid_categories = ['factura', 'libro', 'contrato', 'articulo', 'manual', 
                           'formulario', 'presentacion', 'hoja_calculo', 'imagen', 'otro']
        
        return category if category in valid_categories else "otro"
    except Exception as e:
        print(f"Error en clasificación LLM: {e}")
        return "otro"

def extract_structured_info(text, doc_type, model_name="gemma2:2b-instruct-q4_K_M"):
    """Extrae información estructurada según el tipo de documento usando LLM"""
    if doc_type == "factura":
        prompt = f"""
        Extrae la siguiente información de la factura en formato JSON válido:
        - numero_factura (string)
        - fecha_emision (string en formato YYYY-MM-DD)
        - emisor (string)
        - receptor (string)
        - total (number)
        - moneda (string de 3 caracteres)

        Si no encuentras algún campo, usa null.

        Responde SOLO con el JSON, sin explicaciones ni texto adicional.

        Texto de la factura:
        {text[:2000]}
        """
    elif doc_type == "libro":
        prompt = f"""
        Extrae la siguiente información del libro en formato JSON válido:
        - titulo (string)
        - autor (string)
        - año_publicacion (number)
        - editorial (string)
        - isbn (string o null)

        Si no encuentras algún campo, usa null.

        Responde SOLO con el JSON, sin explicaciones ni texto adicional.

        Texto del libro:
        {text[:2000]}
        """
    else:
        return None
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 500}
        )
        
        json_str = response['message']['content'].strip()
        
        # Limpiar y parsear JSON
        json_str = re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.IGNORECASE)
        json_str = json_str.strip()
        
        data = json.loads(json_str)
        return data
    except Exception as e:
        print(f"Error extrayendo información estructurada: {e}")
        return None

def detect_language(text):
    """Detecta el idioma del texto"""
    try:
        if len(text.strip()) < 10:
            return "en"
        return detect(text)
    except LangDetectException:
        return "en"

def extract_text_from_file(file_path):
    """Extrae texto de diferentes tipos de archivos con mejor manejo de errores"""
    try:
        # Detectar tipo de archivo
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            try:
                mime = magic.Magic(mime=True)
                mime_type = mime.from_file(file_path)
            except:
                # Fallback por extensión
                extension = os.path.splitext(file_path)[1].lower()
                mime_type = {
                    '.txt': 'text/plain',
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.epub': 'application/epub+zip',
                    '.xls': 'application/vnd.ms-excel',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.csv': 'text/csv'
                }.get(extension, 'unknown')
        
        # Extraer texto según el tipo
        if mime_type == 'text/plain':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        elif mime_type == 'application/pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                # Mejorar extracción de texto de PDF
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Limpiar y normalizar texto
                        page_text = ' '.join(page_text.split())  # Eliminar espacios múltiples
                        text += page_text + "\n"
            return text
            
        elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                          'application/msword']:
            return docx2txt.process(file_path)
            
        elif mime_type.startswith('image/'):
            try:
                with Image.open(file_path) as img:
                    return f"Imagen: {os.path.basename(file_path)}, Tamaño: {img.size}, Formato: {img.format}"
            except:
                return f"Archivo de imagen: {os.path.basename(file_path)}"
                
        elif mime_type in ['application/epub+zip']:
            # EPUB necesita manejo especial
            try:
                book = epub.read_epub(file_path)
                text = ""
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        # Usar BeautifulSoup para extraer texto limpio de HTML
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text += soup.get_text() + "\n"
                return text
            except Exception as e:
                print(f"Error procesando EPUB {file_path}: {e}")
                return f"Archivo EPUB: {os.path.basename(file_path)} - Error al extraer texto"
                
        elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            # Excel necesita manejo especial
            try:
                workbook = xlrd.open_workbook(file_path)
                text = ""
                for sheet in workbook.sheets():
                    for row in range(sheet.nrows):
                        for col in range(sheet.ncols):
                            text += str(sheet.cell_value(row, col)) + " "
                        text += "\n"
                return text
            except:
                return f"Archivo Excel: {os.path.basename(file_path)} - No se pudo extraer texto"
        
        elif mime_type == 'text/csv':
            # Procesar archivos CSV
            try:
                text = ""
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        text += ", ".join(row) + "\n"
                return text
            except:
                return f"Archivo CSV: {os.path.basename(file_path)} - No se pudo extraer texto"
                    
        else:
            return f"Tipo de archivo no soportado: {mime_type}"
            
    except Exception as e:
        return f"Error procesando archivo: {str(e)}"

def extract_metadata(file_path):
    """Extrae metadatos avanzados del archivo"""
    try:
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower().replace('.', '')
        mod_time = os.path.getmtime(file_path)
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
        
        metadata = {
            'name': file_name,
            'type': file_extension,
            'size': f"{file_size / 1024:.1f} KB",
            'date': mod_date,
            'file_path': file_path,
            'title': file_name,
            'author': '',
            'creation_date': '',
            'keywords': [],
            'word_count': 0
        }
        
        # Metadatos específicos por tipo de archivo
        if file_path.endswith('.pdf'):
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    metadata['page_count'] = len(pdf_reader.pages)
                    if pdf_reader.metadata:
                        if pdf_reader.metadata.title:
                            metadata['title'] = pdf_reader.metadata.title
                        if pdf_reader.metadata.author:
                            metadata['author'] = pdf_reader.metadata.author
            except:
                pass
                
        elif file_path.endswith(('.doc', '.docx')):
            try:
                doc = Document(file_path)
                metadata['paragraph_count'] = len(doc.paragraphs)
                if doc.core_properties.title:
                    metadata['title'] = doc.core_properties.title
                if doc.core_properties.author:
                    metadata['author'] = doc.core_properties.author
            except:
                pass
                
        elif file_path.endswith('.epub'):
            try:
                book = epub.read_epub(file_path)
                if book.get_metadata('DC', 'title'):
                    metadata['title'] = book.get_metadata('DC', 'title')[0][0]
                if book.get_metadata('DC', 'creator'):
                    metadata['author'] = book.get_metadata('DC', 'creator')[0][0]
            except:
                pass
        
        return metadata
        
    except Exception as e:
        print(f"Error extrayendo metadatos de {file_path}: {e}")
        return {
            'name': os.path.basename(file_path),
            'type': 'unknown',
            'size': 'N/A',
            'date': 'N/A',
            'file_path': file_path,
            'title': os.path.basename(file_path),
            'author': '',
            'creation_date': '',
            'keywords': [],
            'word_count': 0
        }

def extract_cover(file_path, output_path=None, size=(200, 300)):
    """Extrae y redimensiona portada de imágenes, PDFs o EPubs"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            # Extraer primera página del PDF como imagen
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            
            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
            
            pix.save(output_path)
            return output_path
            
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            # Si es una imagen, redimensionarla para miniatura
            img = Image.open(file_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
            
            img.save(output_path, 'PNG')
            return output_path
            
        elif file_extension == '.epub':
            # Intentar extraer la imagen de portada del EPUB
            try:
                book = epub.read_epub(file_path)
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_IMAGE:
                        # Guardar la primera imagen como portada
                        if output_path is None:
                            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                        with open(output_path, 'wb') as f:
                            f.write(item.get_content())
                        return output_path
            except:
                # Si no se puede extraer la portada, usar el icono por defecto
                return get_file_icon_path(file_path)
                
        return get_file_icon_path(file_path)
    except Exception as e:
        print(f"Error extrayendo portada: {str(e)}")
        return get_file_icon_path(file_path)

def get_file_icon_path(file_path):
    """Devuelve la ruta al icono PNG basado en el tipo de archivo"""
    extension = os.path.splitext(file_path)[1].lower().replace('.', '')
    icon_path = os.path.join(ICONS_PATH, f"{extension}.png")
    
    # Si no existe el icono específico, usar el icono por defecto
    if not os.path.exists(icon_path):
        icon_path = os.path.join(ICONS_PATH, "default.png")
    
    return icon_path

def chunk_text(text, chunk_size=600, chunk_overlap=80):
    """Divide texto en chunks inteligentes preservando la estructura"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return text_splitter.split_text(text)

def check_system_resources():
    """Verifica el uso de recursos del sistema y devuelve información"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available': memory.available / (1024 ** 3),  # GB
        'disk_percent': disk.percent
    }

def process_files(uploaded_files, collection_name, base_path="data/chromadb"):
    """Procesa archivos subidos y los prepara para ChromaDB con chunking inteligente"""
    processed_files = []
    
    for uploaded_file in uploaded_files:
        try:
            # Verificar recursos del sistema antes de procesar
            resources = check_system_resources()
            if resources['cpu_percent'] > 85 or resources['memory_percent'] > 85:
                print("⚠️  Sistema sobrecargado, pausando procesamiento...")
                time.sleep(2)
            
            # Guardar archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Extraer texto
            text = extract_text_from_file(tmp_path)
            
            # Solo procesar si se extrajo texto válido
            if text and len(text.strip()) > 20:  # Mínimo 20 caracteres
                language = detect_language(text)
                metadata = extract_metadata(tmp_path)
                
                # Clasificar documento (Paso 1)
                doc_type = classify_document_scikit(text)
                if doc_type == "otro":
                    doc_type = classify_document_llm(text)
                
                metadata['doc_type'] = doc_type
                
                # Dividir en chunks inteligentes
                chunks = chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    processed_files.append({
                        'id': f"{uploaded_file.name}_chunk{i}_{int(time.time())}",
                        'name': f"{uploaded_file.name}_chunk{i}",
                        'content': chunk,
                        'language': language,
                        'size': len(chunk),
                        'metadata': metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_file': uploaded_file.name,
                        'file_path': tmp_path  # Guardar ruta temporal para referencia
                    })
                
                print(f"✅ Procesado: {uploaded_file.name} - {len(chunks)} chunks creados")
            else:
                print(f"⚠️  Saltado: {uploaded_file.name} - texto insuficiente")
            
            # Limpiar archivo temporal
            os.unlink(tmp_path)
            
        except Exception as e:
            print(f"❌ Error procesando {uploaded_file.name}: {str(e)}")
    
    return processed_files

def process_folder_files(file_paths, collection_name, base_path="data/chromadb"):
    """Procesa archivos desde rutas de sistema de archivos con chunking inteligente"""
    processed_files = []
    
    # Inicializar base de datos para documentos estructurados (Paso 2)
    init_db()
    
    for file_path in file_paths:
        try:
            # Verificar recursos del sistema antes de procesar
            resources = check_system_resources()
            if resources['cpu_percent'] > 85 or resources['memory_percent'] > 85:
                print("⚠️  Sistema sobrecargado, pausando procesamiento...")
                time.sleep(2)
            
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                print(f"❌ Archivo no encontrado: {file_path}")
                continue
            
            # Extraer texto
            text = extract_text_from_file(file_path)
            
            # Solo procesar si se extrajo texto válido
            if text and len(text.strip()) > 50:
                language = detect_language(text)
                metadata = extract_metadata(file_path)
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                file_type = os.path.splitext(file_path)[1].lower().replace('.', '')
                
                # Clasificar documento (Paso 1)
                doc_type = classify_document_scikit(text)
                if doc_type == "otro":
                    doc_type = classify_document_llm(text)
                
                metadata['doc_type'] = doc_type
                
                # Guardar información básica en SQL (Paso 2)
                save_document_info(
                    "data/documentos.db", 
                    file_path, 
                    file_name, 
                    file_type, 
                    file_size, 
                    doc_type, 
                    language, 
                    metadata
                )
                
                # Extraer información estructurada para tipos específicos (Paso 2)
                if doc_type in ['factura', 'libro']:
                    structured_info = extract_structured_info(text, doc_type)
                    if structured_info:
                        save_structured_data("data/documentos.db", file_path, structured_info, doc_type)
                
                # Dividir en chunks inteligentes
                chunks = chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    processed_files.append({
                        'id': f"{os.path.basename(file_path)}_chunk{i}_{int(time.time())}",
                        'name': f"{os.path.basename(file_path)}_chunk{i}",
                        'content': chunk,
                        'language': language,
                        'size': len(chunk),
                        'metadata': metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_file': os.path.basename(file_path),
                        'file_path': file_path  # Guardar ruta original
                    })
                
                print(f"✅ Procesado: {os.path.basename(file_path)} - {len(chunks)} chunks creados")
            else:
                print(f"⚠️  Saltado: {os.path.basename(file_path)} - texto insuficiente")
            
        except Exception as e:
            print(f"❌ Error procesando {os.path.basename(file_path)}: {str(e)}")
    
    return processed_files

def scan_folder(folder_path):
    """Escanea una carpeta y devuelve lista de archivos procesables"""
    supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.epub', '.xls', '.xlsx', '.csv']
    processable_files = []
    
    if os.path.exists(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    processable_files.append(os.path.join(root, file))
    
    return processable_files

def generate_file_preview(file_path, max_lines=10):
    """Genera una vista previa del contenido del archivo"""
    try:
        text = extract_text_from_file(file_path)
        if text:
            lines = text.split('\n')
            preview = '\n'.join(lines[:max_lines])
            return preview + '...' if len(lines) > max_lines else preview
        return "No se pudo extraer texto para la vista previa"
    except Exception as e:
        return f"Error generando vista previa: {str(e)}"

def save_processing_log(collection_name, files_processed, output_dir="logs"):
    """Guarda un registro del procesamiento de archivos"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'collection': collection_name,
        'files_processed': files_processed,
        'total_chunks': sum([f.get('total_chunks', 1) for f in files_processed])
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return log_file