# app/chroma_manager.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Optional, Tuple
import uuid
import json
import time
from datetime import datetime
import re
import numpy as np
from functools import lru_cache
import hashlib
import torch
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from rank_bm25 import BM25Okapi

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ChromaManager:
    def __init__(self, data_path="data/chromadb", embedding_model_name=None, device_override=None):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Configuración optimizada para hardware limitado
        self.config = {
            "embedding_model": embedding_model_name or 'sentence-transformers/all-MiniLM-L6-v2',
            "embedding_size": 384,
            "max_cache_size": 500,
            "default_batch_size": 2,
            "gpu_memory_threshold": 1.5 * 1024**3,
            "max_keywords": 5,
            "similarity_threshold": 0.3,
            "rerank_top_n": 5,
            "chunk_size_words": 150,
            "chunk_overlap_words": 30,
            "min_batch_size": 1,
            "max_batch_size": 64,  # Aumentado para mejor uso de GPU
            "db_write_batch_size": 100,  # Escritura en lotes más grandes
            "checkpoint_interval": 1000,  # Checkpoint cada 1000 documentos
            "max_retries": 5,  # Más intentos para errores de disco
            "retry_delay": 30,  # Mayor delay para reintentos
            "hybrid_search_weights": [0.7, 0.3],  # [semántica, keywords]
            "response_quality_threshold": 0.6,  # Umbral de calidad de respuesta
        }
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=data_path
        )
        
        # Detectar y configurar dispositivo
        self.device = device_override or self._detect_best_device()
        
        # Forzar GPU si está disponible
        if torch.cuda.is_available() and self.device != "cuda":
            logger.warning("GPU disponible pero no seleccionada. Forzando uso de GPU...")
            self.device = "cuda"
        
        logger.info(f"🚀 Usando dispositivo: {self.device.upper()}")
        
        # Cargar modelo de embeddings
        self.embedding_model = SentenceTransformer(
            self.config["embedding_model"],
            device=self.device
        )
        
        # Configuración de precisión mixta
        self._setup_mixed_precision()
        
        # Cache para embeddings
        self.embedding_cache = {}
        
        # Cargar stopwords en múltiples idiomas
        self.stop_words = self._load_multilingual_stopwords()
        
        # Estadísticas de rendimiento
        self.performance_stats = {
            "total_chunks_processed": 0,
            "total_time": 0,
            "avg_chunks_per_second": 0
        }
        
        # Nuevos atributos para BM25 e índice híbrido
        self.bm25_index = None
        self.processed_chunks = []
        self.raw_chunks = []
        
        # Inicializar índice BM25 si hay datos existentes
        self._initialize_bm25_index()
        
        # Habilitar modo de alto rendimiento si hay GPU
        if self.device == 'cuda':
            self.enable_high_performance_mode()
            
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocesa texto para BM25 con mejor manejo de casos edge"""
        if not text or not isinstance(text, str):
            return []
    
        try:
            # Limpieza básica del texto
            text = text.lower().strip()
            if not text:
                return []
        
            # Tokenización
            words = word_tokenize(text)
        
            # Filtrar palabras válidas
            valid_words = []
            for word in words:
                # Mantener palabras con letras y algunos símboles útiles
                if len(word) > 1 and any(c.isalpha() for c in word):
                    # Limpiar palabra de símbolos no deseados
                    cleaned_word = ''.join(c for c in word if c.isalnum() or c in ['-', '_'])
                    if cleaned_word and cleaned_word not in self.stop_words:
                        valid_words.append(cleaned_word)
        
            return valid_words
        
        except Exception as e:
            logger.warning(f"Error en preprocesamiento de texto: {e}")
            return []

    def _setup_mixed_precision(self):
        """Configura precisión mixta para ahorrar memoria en GPU"""
        if self.device == 'cuda':
            try:
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')  # Cambiado a 'high' para mejor rendimiento
                logger.info("✅ Precisión mixta habilitada para GPU")
            except Exception as e:
                logger.warning(f"No se pudo configurar precisión mixta: {e}")

    def _load_multilingual_stopwords(self):
        """Carga stopwords en múltiples idiomas"""
        try:
            languages = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese']
            stopwords_set = set()
        
            for lang in languages:
                try:
                    stopwords_set.update(stopwords.words(lang))
                except LookupError:
                    try:
                        nltk.download('stopwords', quiet=True)
                        stopwords_set.update(stopwords.words(lang))
                    except:
                        logger.warning(f"No se pudieron cargar stopwords para {lang}")
                        continue
        
            logger.info(f"✅ Stopwords cargados para {len(languages)} idiomas")
            return stopwords_set
        
        except Exception as e:
            logger.error(f"Error cargando stopwords multilingües: {e}")
            return set()
    
    def _detect_best_device(self):
        """Detecta el mejor dispositivo disponible"""
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"📊 Memoria GPU disponible: {gpu_memory/1024**3:.1f}GB")
                
                if gpu_memory >= 1.5 * 1024**3:
                    return 'cuda'
                else:
                    logger.warning(f"⚠️  GPU con poca memoria ({gpu_memory/1024**3:.1f}GB), usando CPU")
                    return 'cpu'
            except Exception as e:
                logger.error(f"Error detectando GPU: {e}")
                return 'cpu'
        return 'cpu'
    
    def _optimize_batch_size(self, texts):
        """Optimiza el tamaño del lote dinámicamente basado en memoria GPU disponible"""
        if self.device != 'cuda':
            return self.config["default_batch_size"]
        
        try:
            # Obtener memoria libre de GPU
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            free_memory = total_memory - allocated_memory
            
            # Calcular tamaño de lote basado en memoria libre
            if free_memory > 1.5 * 1024**3:  # Más de 1.5GB libre
                return min(self.config["max_batch_size"], 32)
            elif free_memory > 1 * 1024**3:   # Más de 1GB libre
                return min(self.config["max_batch_size"], 16)
            elif free_memory > 0.5 * 1024**3: # Más de 0.5GB libre
                return min(self.config["max_batch_size"], 8)
            else:
                return self.config["min_batch_size"]
                
        except Exception as e:
            logger.warning(f"Error optimizando batch size: {e}, usando valor por defecto")
            return self.config["default_batch_size"]

    def enable_disk_safe_mode(self):
        logger.info("💾 Activando modo seguro para disco")
        self.config.update({
            "default_batch_size": 4,
            "max_batch_size": 8,
            "db_write_batch_size": 50,
            "retry_delay": 30,  # Delay más largo para errores de disco
        })
    
    def _safe_encode(self, texts):
        """Codificación segura con manejo de errores and optimización de memoria"""
        if not texts:
            return np.array([])
            
        start_time = time.time()
        optimal_batch_size = self._optimize_batch_size(texts)
        results = []
        
        for i in range(0, len(texts), optimal_batch_size):
            batch_texts = texts[i:i + optimal_batch_size]
            
            try:
                # Usar autocast para precisión mixta en GPU
                if self.device == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        batch_result = self.embedding_model.encode(
                            batch_texts, 
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            normalize_embeddings=True
                        )
                else:
                    batch_result = self.embedding_model.encode(
                        batch_texts, 
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                
                results.append(batch_result)
                
                # Liberar memoria de GPU después de cada lote
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"⚠️  OOM con batch size {optimal_batch_size}, reduciendo...")
                    reduced_batch_size = max(1, optimal_batch_size // 2)
                    for j in range(0, len(batch_texts), reduced_batch_size):
                        retry_batch = batch_texts[j:j + reduced_batch_size]
                        retry_result = self.embedding_model.encode(
                            retry_batch,
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            normalize_embeddings=True
                        )
                        results.append(retry_result)
                else:
                    raise e
        
        if results:
            final_result = np.vstack(results)
            processing_time = time.time() - start_time
            self._update_performance_stats(len(texts), processing_time)
            return final_result
        else:
            return np.array([])

    def _generate_embedding(self, text: str) -> List[float]:
        """Genera el embedding para un texto dado utilizando el modelo de SentenceTransformer"""
        try:
            # Usar el mismo método que _safe_encode pero para un solo texto
            if self.device == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    embedding = self.embedding_model.encode(
                        [text],
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
            else:
                embedding = self.embedding_model.encode(
                    [text],
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            
            return embedding[0].tolist()  # Devolver el primer (y único) embedding como lista
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return []

    def _optimize_chunking(self, documents: List[Dict]) -> List[Dict]:
        """Optimiza el troceado de documentos para mejorar la eficiencia"""
        optimized_docs = []
        for doc in documents:
            content = doc.get('content', '')
            
            # Chunks más pequeños para hardware limitado
            if len(content) > 500:
                chunk_size = 100
                overlap = 20
            else:
                chunk_size = self.config["chunk_size_words"]
                overlap = self.config["chunk_overlap_words"]
        
            chunks = self._split_text(content, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                optimized_doc = doc.copy()
                optimized_doc['content'] = chunk
                optimized_doc['chunk_index'] = i
                optimized_doc['total_chunks'] = len(chunks)
                optimized_docs.append(optimized_doc)
        return optimized_docs

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Divide el texto en chunks con solapamiento"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks

    def _generate_deterministic_id(self, doc: Dict) -> str:
        """Genera un ID determinístico único basado en el contenido y metadatos"""
        # Información base para el ID
        file_path = doc.get('file_path', 'unknown')
        chunk_index = doc.get('chunk_index', 0)
        content_preview = doc.get('content', '')[:500]  # Primeros 500 caracteres
        document_id = doc.get('id', str(uuid.uuid4()))
        # Crear una cadena única basada en file_path y chunk_index
        # Incluir timestamp para mayor unicidad
        timestamp = int(time.time() * 1000)  # Milisegundos para mayor precisión
        
        # unique_string = f"{file_path}_{chunk_index}_{timestamp}_{content_preview}"

        # Generar hash MD5 (más rápido y suficiente para este propósito)    
        # Crear una cadena única
        if file_path:
            unique_string = f"{file_path}_{chunk_index}_{document_id}_{timestamp}_{content_preview}"
        else:
            unique_string = f"{chunk_index}_{document_id}_{timestamp}_{content_preview}"

        
        # Generar hash MD5 (más rápido y suficiente para este propósito)
        return hashlib.sha256(unique_string.encode()).hexdigest()

    def _remove_existing_documents(self, collection, documents: List[Dict]) -> List[Dict]:
        """Filtra documentos que ya existen en la colección para evitar duplicados"""
        unique_docs = []
        existing_ids = set()
    
        try:
            # Obtener todos los IDs existentes en la colección
            existing_records = collection.get()
            if existing_records['ids']:
                existing_ids = set(existing_records['ids'])
        except Exception as e:
            logger.warning(f"Error obteniendo documentos existentes: {e}")
    
        # Filtrar documentos nuevos
        for doc in documents:
            doc_id = self._generate_deterministic_id(doc)
            if doc_id not in existing_ids:
                unique_docs.append(doc)
            else:
                logger.debug(f"Documento duplicado omitido: {doc_id}")
    
        logger.info(f"📊 Documentos únicos a agregar: {len(unique_docs)}/{len(documents)}")
        return unique_docs
    
    def _update_performance_stats(self, chunks_processed, processing_time):
        """Actualiza las estadísticas de rendimiento"""
        self.performance_stats["total_chunks_processed"] += chunks_processed
        self.performance_stats["total_time"] += processing_time
        
        if self.performance_stats["total_time"] > 0:
            self.performance_stats["avg_chunks_per_second"] = (
                self.performance_stats["total_chunks_processed"] / 
                self.performance_stats["total_time"]
            )

    def _check_system_resources(self):
        """Verifica el estado de los recursos del sistema con umbrales optimizados para GPU"""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage(self.data_path).percent,
            'gpu_percent': 0,
            'gpu_memory': 0,
            'overloaded': False
        }
        
        # Umbrales optimizados para GPU
        cpu_threshold = 90  # Aumentado para permitir más uso de CPU
        memory_threshold = 85
        gpu_threshold = 90  # Aumentado para permitir más uso de GPU

        # Verificar uso de disco
        if resources['disk_usage'] > 90:
            logger.warning(f"⚠️  Disco casi lleno: {resources['disk_usage']}%")
            resources['overloaded'] = True
    
        # Verificar CPU y RAM
        if resources['cpu_percent'] > 85 or resources['memory_percent'] > 85:
            resources['overloaded'] = True

        try:
            if self.device == 'cuda' and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                resources['gpu_memory'] = (allocated_memory / total_memory) * 100
                
                # Calcular memoria libre de GPU
                free_memory = total_memory - allocated_memory
                
                # No marcar como sobrecargado si la GPU tiene suficiente memoria libre
                if free_memory > 500 * 1024**2:  # 500MB libres
                    resources['overloaded'] = False
                else:
                    # Verificar sobrecarga solo si la GPU está limitada
                    if (resources['cpu_percent'] > cpu_threshold or 
                        resources['memory_percent'] > memory_threshold or
                        resources['gpu_memory'] > gpu_threshold):
                        resources['overloaded'] = True
            else:
                # Verificar sobrecarga solo con CPU/RAM
                if (resources['cpu_percent'] > cpu_threshold or 
                    resources['memory_percent'] > memory_threshold):
                    resources['overloaded'] = True
                    
        except Exception as e:
            logger.warning(f"Error verificando recursos del sistema: {e}")
            # En caso de error, asumir sobrecarga para ser conservador
            resources['overloaded'] = True
            
        return resources

    def _adjust_batch_size(self, resources, current_batch_size):
        """Ajusta el tamaño del lote basado principalmente en la memoria de GPU"""
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                free_memory = total_memory - allocated_memory
                
                # Aumentar batch size si hay memoria libre disponible
                if free_memory > 1 * 1024**3:  # Más de 1GB libre
                    return min(self.config["max_batch_size"], current_batch_size * 2)
                # Reducir batch size si la memoria libre es crítica
                elif free_memory < 200 * 1024**2:  # Menos de 200MB libre
                    return max(self.config["min_batch_size"], current_batch_size // 2)
            except Exception as e:
                logger.warning(f"Error ajustando batch size por GPU: {e}")
        
        # Lógica de respaldo para CPU/RAM
        if resources['cpu_percent'] > 85 or resources['memory_percent'] > 85:
            return max(self.config["min_batch_size"], current_batch_size // 2)
            
        if resources['cpu_percent'] < 60 and resources['memory_percent'] < 70:
            return min(self.config["max_batch_size"], current_batch_size * 2)
            
        return current_batch_size
    
    def _force_memory_cleanup(self):
        """Limpia memoria intensivamente"""
        import gc
    
        # Liberar memoria de Python
        gc.collect()
    
        # Limpiar caché de embeddings
        self.embedding_cache.clear()
    
        # Limpiar memoria de GPU si está disponible
        if self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Liberar memoria no utilizada
                torch.cuda.memory._dump_snapshot()
            except Exception as e:
                logger.warning(f"No se pudo limpiar memoria GPU: {e}")
    
        # Pausa más larga para permitir la liberación de memoria
        time.sleep(5)  # Aumenta a 5 segundos
    
    def _log_gpu_usage(self):
        """Registra el uso de la GPU si está disponible"""
        if self.device == 'cuda':
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 'N/A'
                
                logger.info(f"📊 Uso GPU - Memoria asignada: {gpu_memory_allocated:.2f}GB, "
                          f"Memoria reservada: {gpu_memory_cached:.2f}GB, "
                          f"Memoria total: {gpu_memory_total:.2f}GB, "
                          f"Utilización: {gpu_utilization}%")
            except Exception as e:
                logger.warning(f"No se pudo obtener información de GPU: {e}")
    
    def enable_low_power_mode(self):
        """Habilita el modo de bajo consumo para hardware limitado"""
        logger.info("🔋 Activando modo de bajo consumo")
        
        # Configuración optimizada para bajo consumo
        self.config.update({
            "default_batch_size": 8,    # antes 1
            "max_batch_size": 16,   # antes 2
            "chunk_size_words": 100,
            "chunk_overlap_words": 20,
            "max_cache_size": 100,
            "max_keywords": 3,
            "rerank_top_n": 3,
            "db_write_batch_size": 100,  # Nuevo: lote específico para escritura en DB
            "checkpoint_interval": 500,  # Guardar checkpoint cada 500 documentos
        })
        
        # Limpiar cachés
        self._force_memory_cleanup()
        
        # Si está usando GPU, considerar cambiar a CPU
        if self.device == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            if gpu_memory > 1.0:  # Si usa más de 1GB
                logger.warning("⚠️  Memoria GPU limitada, considerando cambio a CPU")
    
    def enable_high_performance_mode(self):
        """Habilita el modo de alto rendimiento para maximizar el uso de GPU"""
        logger.info("🚀 Activando modo de alto rendimiento")
        
        # Configuración optimizada para GPU
        self.config.update({
            "default_batch_size": 16,
            "max_batch_size": 64,
            "chunk_size_words": 200,
            "chunk_overlap_words": 50,
            "max_cache_size": 1000,
        })
        
        # Forzar limpieza de memoria
        self._force_memory_cleanup()
    
    def add_documents(self, collection_name: str, documents: List[Dict]):
        """Añade documentos to the collection with optimized processing"""

        # Añadir al inicio del método
        total_added = 0
        processed_ids = []
        # Dentro del bucle principal, después de procesar cada lote
        if total_added > 0:
            logger.info(f"📊 {total_added} documentos agregados en el lote")
            self._log_gpu_usage()
            self._force_memory_cleanup()
            total_added = 0  # Reiniciar contador

        if not documents:
            logger.warning("No se recibieron documentos para procesar")
            return 0
            
        # Optimizar el troceado de documentos
        optimized_docs = self._optimize_chunking(documents)
        
        # Obtener la colección
        collection = self.create_collection(collection_name)
        
        # Filtrar documentos existentes para evitar duplicados
        optimized_docs = self._remove_existing_documents(collection, optimized_docs)
        
        if not optimized_docs:
            logger.info("✅ Todos los documentos ya existen en la colección")
            return 0
            
        # Verificar recursos del sistema
        resources = self._check_system_resources()
        
        total_added = 0
        start_time = time.time()
        batch_size = self._adjust_batch_size(resources, self.config["default_batch_size"])
        
        logger.info(f"📊 Procesando {len(optimized_docs)} chunks optimizados...")
        logger.info(f"🔧 Usando dispositivo: {self.device.upper()}")
        logger.info(f"⚡ Batch size inicial: {batch_size}")
        
        # Conjunto para trackear IDs ya procesados
        processed_ids = set()
        
        retry_count = 0
        max_retries = self.config["max_retries"]
        
        i = 0
        while i < len(optimized_docs) and retry_count < max_retries:
            # Verificar recursos del sistema
            resources = self._check_system_resources()
            batch_size = self._adjust_batch_size(resources, batch_size)
            
            # Pausar si el sistema está sobrecargado
            if resources['overloaded']:
                logger.warning(f"⚠️  Sistema sobrecargado (CPU: {resources['cpu_percent']}%, RAM: {resources['memory_percent']}%), pausando...")
                time.sleep(self.config["retry_delay"])
                continue
            
            # Procesar lote
            batch = optimized_docs[i:i + batch_size]
            
            try:
                # PRIMERO: Filtrar documentos duplicados dentro del lote actual
                filtered_batch = []
                batch_ids_to_process = []
                
                for doc in batch:
                    doc_id = self._generate_deterministic_id(doc)
                    
                    # Saltar si este ID ya fue procesado
                    if doc_id in processed_ids:
                        logger.debug(f"⚠️  Saltando documento duplicado: {doc_id}")
                        continue
                    
                    filtered_batch.append(doc)
                    batch_ids_to_process.append(doc_id)
                
                # Si no hay documentos válidos en the lote, continuar
                if not filtered_batch:
                    i += batch_size
                    continue
                    
                # GENERAR EMBEDDINGS solo para documentos no duplicados
                batch_texts = [doc['content'] for doc in filtered_batch]
                batch_embeddings = self._safe_encode(batch_texts)
                
                # Verificar que los embeddings coincidan con los documentos filtrados
                if batch_embeddings is None or len(batch_embeddings) != len(filtered_batch):
                    logger.error(f"❌ Error de embeddings: {len(batch_embeddings)} != {len(filtered_batch)}")
                    batch_size = max(1, batch_size // 2)
                    continue
                
                # PREPARAR METADATOS para documentos no duplicados
                batch_metadatas = []
                batch_documents = []
                
                for doc in filtered_batch:
                    enhanced_metadata = self._extract_enhanced_metadata(doc)
                    batch_metadatas.append(enhanced_metadata)
                    batch_documents.append(doc['content'])
                
                # Añadir lote a la colección
                collection.add(
                    ids=batch_ids_to_process,
                    embeddings=batch_embeddings.tolist(),
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
                # Actualizar IDs procesados
                processed_ids.update(batch_ids_to_process)
                total_added += len(batch_ids_to_process)
                i += batch_size
                
                # Reiniciar contador de reintentos si el lote tiene éxito
                retry_count = 0
                
                # Mostrar progreso cada 50 chunks para evitar duplicados en logging
                if total_added % 50 == 0:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        docs_per_second = total_added / elapsed_time
                        remaining = len(optimized_docs) - i
                        eta_seconds = remaining / docs_per_second if docs_per_second > 0 else 0
                        
                        logger.info(f"✅ Procesados: {total_added}/{len(optimized_docs)} chunks "
                                  f"({total_added/len(optimized_docs)*100:.1f}%), "
                                  f"Velocidad: {docs_per_second:.1f} chunks/seg")
                
                # Pequeña pausa para evitar sobrecarga
                time.sleep(0.1)
                
                # Registrar uso de GPU periódicamente
                if total_added % 50 == 0:
                    self._log_gpu_usage()
                
            except Exception as e:
                logger.error(f"❌ Error procesando lote: {e}")
                retry_count += 1

                if "disk I/O error" in str(e):
                    logger.warning("Error de I/O de disco, reintentando con delay...")
                    time.sleep(self.config["retry_delay"])
                    self._force_memory_cleanup()

                    # Reducir batch size más agresivamente
                    batch_size = max(1, batch_size // 2)
                    logger.info(f"🔧 Reduciendo tamaño de lote a {batch_size}")        
                    continue

                if "CUDA out of memory" in str(e):
                    logger.warning("Error de memoria de GPU, reintentando con ajuste de batch size...")
                    batch_size = max(1, batch_size // 2)
                    self._force_memory_cleanup()
                    continue                   
                
                if retry_count >= max_retries:
                    logger.error(f"🚨 Demasiados errores consecutivos. Abortando procesamiento.")
                    break
                    
                # Reducir tamaño de lote más agresivamente
                batch_size = max(1, batch_size // 2)
                logger.info(f"🔧 Reduciendo tamaño de lote a {batch_size} (reintento {retry_count}/{max_retries})")
                
                # Pausa más larga para permitir que el sistema se recupere
                time.sleep(self.config["retry_delay"])
                
                # Limpiar memoria intensivamente
                self._force_memory_cleanup()
        
        # Guardar metadata de la colección
        self._save_collection_metadata(collection_name, total_added)
        
        # Actualizar índice BM25 con los nuevos documentos
        self._update_bm25_index(optimized_docs)
        
        # Limpiar memoria final
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        final_speed = total_added / total_time if total_time > 0 else 0
        
        # Log final sin duplicados
        logger.info(f"🎉 Procesamiento completado: {total_added} chunks en {total_time/60:.1f} minutos "
                  f"({final_speed:.1f} chunks/segundo)")
        
        return total_added

    def _initialize_bm25_index(self):
        """Inicializa el índice BM25 con validaciones mejoradas"""
        try:
            collections = self.client.list_collections()
            if not collections:
                logger.info("📊 No hay colecciones existentes para inicializar BM25")
                return

            collection = self.client.get_collection(collections[0].name)
            results = collection.get()

            if results and results.get('documents'):
                self.raw_chunks = []
                self.processed_chunks = []
            
                for doc_content in results['documents']:
                    if doc_content and doc_content.strip():
                        processed = self._preprocess_text(doc_content)
                        if processed:
                            self.raw_chunks.append(doc_content)
                            self.processed_chunks.append(processed)

                # Verificar que tenemos documentos válidos
                if len(self.processed_chunks) >= 2:
                    self.bm25_index = BM25Okapi(self.processed_chunks)
                    logger.info(f"✅ Índice BM25 inicializado con {len(self.processed_chunks)} documentos")
                else:
                    logger.warning("⚠️ No hay suficientes documentos válidos para inicializar BM25")
            else:
                logger.info("📊 No hay documentos en la colección para inicializar BM25")
            
        except ZeroDivisionError:
            logger.warning("⚠️ División por cero al inicializar BM25 - corpus insuficiente")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar BM25: {e}")

    def _update_bm25_index(self, documents: List[Dict]):
        """Actualiza el índice BM25 con nuevos documentos y maneja division by zero"""
        try:
            # Verificar que hay documentos para procesar
            if not documents:
                logger.warning("⚠️ No hay documentos para actualizar BM25")
                return

            new_raw_chunks = []
            new_processed_chunks = []
            skipped_count = 0

            for doc in documents:
                content = doc.get('content', '')
                if content and content.strip():
                    processed_content = self._preprocess_text(content)
                    if processed_content:  # Solo añadir si hay tokens válidos
                        new_raw_chunks.append(content)
                        new_processed_chunks.append(processed_content)
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1

            # Log de documentos omitidos
            if skipped_count > 0:
                logger.info(f"⚠️ Se omitieron {skipped_count} documentos vacíos o inválidos para BM25")

            # Solo actualizar si hay nuevos documentos válidos
            if new_processed_chunks:
                # Añadir a los chunks existentes
                self.raw_chunks.extend(new_raw_chunks)
                self.processed_chunks.extend(new_processed_chunks)
            
                # Verificar que tenemos documentos suficientes
                if len(self.processed_chunks) >= 2:  # Mínimo 2 documentos para BM25
                    try:
                        self.bm25_index = BM25Okapi(self.processed_chunks)
                        logger.info(f"✅ Índice BM25 actualizado. Total documentos: {len(self.processed_chunks)}")
                    except ZeroDivisionError:
                        logger.warning("⚠️ BM25 no pudo inicializarse (división por cero)")
                    except Exception as e:
                        logger.error(f"❌ Error creando índice BM25: {e}")
                else:
                    logger.warning(f"⚠️ No hay suficientes documentos para BM25 ({len(self.processed_chunks)})")
            else:
                logger.warning("⚠️ No hay documentos válidos para actualizar BM25")
            
        except Exception as e:
            logger.error(f"Error actualizando índice BM25: {e}")

    def add_documents_with_recovery(self, collection_name: str, documents: List[Dict], 
                                  recovery_file: str = "processing_checkpoint.json"):
        """Añade documentos con capacidad de recuperación de errores"""
        # Cargar estado previo si existe
        if os.path.exists(recovery_file):
            try:
                with open(recovery_file, 'r') as f:
                    recovery_data = json.load(f)
                    processed_ids = set(recovery_data.get('processed_ids', []))
                    start_index = recovery_data.get('last_index', 0)
                logger.info(f"♻️  Reanudando procesamiento desde índice {start_index}")
            except:
                processed_ids = set()
                start_index = 0
        else:
            processed_ids = set()
            start_index = 0
        
        # Procesar documentos
        try:
            # Crear una copia de los documentos para procesar
            docs_to_process = documents[start_index:]
            
            # Procesar normalmente
            result = self.add_documents(collection_name, docs_to_process)
            
            # Eliminar archivo de recuperación si todo fue bien
            if os.path.exists(recovery_file):
                os.remove(recovery_file)
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Error durante el procesamiento: {e}")
            
            # Guardar estado actual para recuperación
            recovery_data = {
                'collection_name': collection_name,
                'processed_ids': list(processed_ids),
                'last_index': start_index,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(recovery_file, 'w') as f:
                json.dump(recovery_data, f)
                
            logger.info(f"💾 Estado guardado en {recovery_file} para recuperación")
            
            raise e

    def _extract_enhanced_metadata(self, doc: Dict) -> Dict:
        """Extrae metadatos mejorados del documento"""
        metadata = {
            'name': doc.get('name', ''),
            'language': doc.get('language', ''),
            'size': doc.get('size', 0),
            'source': 'local',
            'original_file': doc.get('original_file', ''),
            'chunk_index': doc.get('chunk_index', 0),
            'total_chunks': doc.get('total_chunks', 1),
            'processing_date': datetime.now().isoformat(),
            'file_path': doc.get('file_path', ''),
            'is_chunk': doc.get('is_chunk', False)
        }
        
        # Añadir metadatos adicionales si existen
        if 'metadata' in doc and isinstance(doc['metadata'], dict):
            metadata.update(doc['metadata'])
        
        # Extraer y añadir keywords semánticas
        content = doc.get('content', '')
        keywords = self._extract_semantic_keywords(content)
        metadata['keywords'] = ', '.join(keywords[:self.config["max_keywords"]])
        
        # Extraer categoría semántica mejorada
        metadata['category'] = self._detect_semantic_category(content, keywords)
        
        # Añadir resumen del contenido si es largo
        if len(content) > 500:
            metadata['summary'] = self._generate_content_summary(content)
        
        return metadata
    
    def _extract_semantic_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave semánticamente relevantes usando TF-IDF mejorado"""
        try:
            # Tokenizar y limpiar texto
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
            
            if not words:
                return []
            
            # Calcular frecuencia de términos
            word_freq = Counter(words)
            
            # Usar TF-IDF para identificar términos importantes
            vectorizer = TfidfVectorizer(max_features=20, stop_words=list(self.stop_words))
            try:
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                
                # Combinar frecuencia y TF-IDF para puntuación final
                keywords = []
                for word in feature_names:
                    if word in word_freq:
                        # Puntuación que combina TF-IDF y frecuencia
                        score = tfidf_matrix[0, vectorizer.vocabulary_[word]] * word_freq[word]
                        keywords.append((word, score))
                
                # Ordenar por puntuación
                keywords.sort(key=lambda x: x[1], reverse=True)
                return [word for word, score in keywords[:self.config["max_keywords"]]]
                
            except Exception:
                # Fallback a frecuencia simple si TF-IDF falla
                return [word for word, count in word_freq.most_common(self.config["max_keywords"])]
            
        except Exception:
            return []
    
    def _detect_semantic_category(self, text: str, keywords: List[str]) -> str:
        """Detecta la categoría semántica del contenido usando keywords y contexto"""
        text_lower = text.lower()
        
        # Categorías expandidas con más términos
        categories = {
            'technical': ['código', 'programación', 'tecnología', 'software', 'hardware', 'algoritmo', 
                         'función', 'variable', 'clase', 'objeto', 'python', 'java', 'javascript', 
                         'html', 'css', 'sql', 'base de datos', 'api', 'web', 'aplicación', 'desarrollo'],
            'culinary': ['receta', 'cocina', 'ingrediente', 'chef', 'comida', 'plato', 'aperitivo', 
                        'entrante', 'principal', 'postre', 'bebida', 'horno', 'microondas', 'sartén', 
                        'olla', 'cocinar', 'hornear', 'hervir', 'freír', 'asado', 'guiso', 'ensalada'],
            'academic': ['estudio', 'investigación', 'universidad', 'tesis', 'paper', 'artículo', 
                        'científico', 'publicación', 'revista', 'conferencia', 'hipótesis', 'método'],
            'business': ['empresa', 'negocio', 'mercado', 'finanzas', 'inversión', 'marketing', 
                        'ventas', 'cliente', 'producto', 'servicio', 'estrategia', 'competencia'],
            'literary': ['novela', 'poesía', 'literatura', 'capítulo', 'personaje', 'trama', 
                        'escenario', 'diálogo', 'autor', 'escritor', 'poeta', 'cuento', 'historia'],
            'medical': ['enfermedad', 'paciente', 'médico', 'hospital', 'clínica', 'diagnóstico', 
                       'tratamiento', 'medicamento', 'fármaco', 'dosis', 'inyección', 'cirugía'],
            'legal': ['ley', 'abogado', 'juez', 'tribunal', 'juicio', 'demanda', 'demandado', 
                     'demandante', 'testigo', 'prueba', 'veredicto', 'sentencia', 'cárcel'],
            'general': []
        }
        
        # Calcular puntuación para cada categoría
        scores = {category: 0 for category in categories}
        
        # Puntuación basada en keywords
        for keyword in keywords:
            for category, terms in categories.items():
                if keyword in terms:
                    scores[category] += 2
        
        # Puntuación basada en contenido
        for category, terms in categories.items():
            for term in terms:
                if term in text_lower:
                    scores[category] += 1
        
        # Encontrar categoría con mayor puntuación
        best_category, best_score = max(scores.items(), key=lambda x: x[1])
        
        return best_category if best_score > 0 else 'general'
    
    def _generate_content_summary(self, text: str, max_sentences: int = 2) -> str:
        """Genera un resumen breve del contenido"""
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return text
            
            # Seleccionar las primeras oraciones como resumen
            return ' '.join(sentences[:max_sentences])
        except:
            return text[:200] + "..." if len(text) > 200 else text
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Crea una nueva colección con metadata mejorada"""
        default_metadata = {
            "hnsw:space": "cosine",
            "description": f"Colección para {collection_name}",
            "created_at": datetime.now().isoformat(),
            'embedding_model': self.config["embedding_model"],
            'embedding_size': self.config["embedding_size"],
            'chunk_size': self.config["chunk_size_words"],
            'chunk_overlap': self.config["chunk_overlap_words"]
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=default_metadata
        )
    
    def _save_collection_metadata(self, collection_name: str, document_count: int):
        """Guarda metadata de la colección de forma segura"""
        metadata_path = os.path.join(self.data_path, "collections_metadata.json")
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        
        current_time = datetime.now().isoformat()
        
        if collection_name not in metadata:
            metadata[collection_name] = {
                'document_count': document_count,
                'created_at': current_time,
                'updated_at': current_time,
                'chunk_count': document_count
            }
        else:
            metadata[collection_name].update({
                'document_count': document_count,
                'updated_at': current_time,
                'chunk_count': document_count
            })
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")
    
    def hybrid_search(self, query: str, collection_name: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """Búsqueda híbrida que combina embeddings semánticos y BM25"""
        try:
            collection = self.client.get_collection(collection_name)
        
            # Búsqueda semántica
            semantic_results = collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'distances']
            )
        
            semantic_docs = semantic_results['documents'][0]
            semantic_scores = [1 - dist for dist in semantic_results['distances'][0]] if semantic_results['distances'] else [1.0] * len(semantic_docs)
        
            # Búsqueda por keywords con BM25
            query_tokens = self._preprocess_text(query)
            bm25_scores = self.bm25_index.get_scores(query_tokens)
        
            # Convertir a array de NumPy si no lo es
            import numpy as np
            if not isinstance(bm25_scores, np.ndarray):
                bm25_scores = np.array(bm25_scores)
        
            # Manejar caso de scores vacíos o todos cero
            if len(bm25_scores) == 0 or np.all(bm25_scores == 0):
                max_bm25 = 1.0
            else:
                max_bm25 = np.max(bm25_scores)
        
            # Combinar resultados
            combined_results = {}
            semantic_weight, keyword_weight = self.config["hybrid_search_weights"]
        
            # Crear un mapeo de documentos a scores semánticos
            semantic_map = {doc: score for doc, score in zip(semantic_docs, semantic_scores)}
        
            for i, doc in enumerate(self.raw_chunks):
                # Normalizar scores BM25
                normalized_bm25 = bm25_scores[i] / max_bm25
            
                # Obtener score semántico (0 si no está en los resultados semánticos)
                semantic_score = semantic_map.get(doc, 0.0)
            
                # Score combinado
                combined_score = (semantic_weight * semantic_score) + (keyword_weight * normalized_bm25)
                combined_results[doc] = combined_score
        
            # Ordenar por score combinado y devolver top k
            sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in sorted_results[:k]]
            top_scores = [score for doc, score in sorted_results[:k]]
        
            return top_docs, top_scores
        
        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {e}")
            # Fallback a búsqueda semántica
            return self.semantic_search(query, collection_name, k)
            
    def semantic_search(self, query: str, collection_name: str, k: int = 10):
        """Búsqueda puramente semántica (tu método actual)"""
        collection = self.client.get_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'distances']
        )
        return results['documents'][0], [1 - dist for dist in results['distances'][0]]

    def evaluate_response_quality(self, response: str, context_chunks: List[str], threshold: float = None) -> bool:
        """Evalúa si la respuesta es consistente con el contexto usando embeddings"""
        if threshold is None:
            threshold = self.config["response_quality_threshold"]
            
        try:
            if not response or not context_chunks:
                return False
                
            # Calcular embedding de la respuesta
            resp_embedding = self.embedding_model.encode(response, convert_to_tensor=True)
            
            # Calcular embeddings de los chunks de contexto
            context_embeddings = self.embedding_model.encode(context_chunks, convert_to_tensor=True)
            
            # Calcular similitud coseno
            from sentence_transformers import util
            similarities = util.pytorch_cos_sim(resp_embedding, context_embeddings)[0]
            max_similarity = max(similarities).item()
            
            return max_similarity >= threshold
            
        except Exception as e:
            logger.error(f"Error evaluando calidad de respuesta: {e}")
            return True  # Fallback: asumir que es válida

    def generate_validation_prompt(self, query: str, context: List[str], response: str) -> str:
        """Genera un prompt para validar la respuesta contra el contexto"""
        context_str = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        return f"""
        Evalúa si la siguiente respuesta es coherente con el contexto proporcionado y responde ÚNICAMENTE con 'True' o 'False'.
        
        CONTEXTO:
        {context_str}
        
        PREGUNTA: {query}
        RESPUESTA: {response}
        
        ¿La respuesta se basa ÚNICAMENTE en el contexto proporcionado y es factualmente correcta?
        """

    def search(self, collection_name: str, query: str, n_results: int = 5, 
               threshold: float = None, use_reranking: bool = True, use_hybrid: bool = True):
        """Busca documentos similares con filtrado de relevancia mejorado"""
        try:
            # Usar búsqueda híbrida o semántica según configuración
            if use_hybrid and self.bm25_index is not None:
                context_chunks, scores = self.hybrid_search(query, collection_name, n_results * 3)
            else:
                context_chunks, scores = self.semantic_search(query, collection_name, n_results * 3)
            
            if not context_chunks:
                return {'documents': [], 'metadatas': [], 'distances': []}
            
            # Obtener metadatos para los chunks
            collection = self.client.get_collection(collection_name)
            results = collection.get()
            metadatas_by_content = {}
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    if i < len(results['documents']):
                        content = results['documents'][i]
                        metadatas_by_content[content] = metadata
            
            # Filtrar y rerankear resultados
            filtered_results = self._filter_and_rerank_results(
                {'documents': [context_chunks], 'metadatas': [[metadatas_by_content.get(doc, {}) for doc in context_chunks]], 'distances': [[1 - score for score in scores]]},
                query, query, threshold or self.config["similarity_threshold"], use_reranking
            )
            
            # Limitar al número de resultados solicitados
            final_results = {
                'documents': [filtered_results['documents'][:n_results]],
                'metadatas': [filtered_results['metadatas'][:n_results]],
                'distances': [filtered_results['distances'][:n_results]]
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def _enhance_query(self, query: str) -> str:
        """Mejora la query para búsquedas más precisas usando expansión de términos"""
        query = query.lower().strip()
        
        if not query:
            return query
        
        # Diccionario de expansión de términos por categoría
        query_expansions = {
            'cómo': ['cómo', 'método', 'procedimiento', 'pasos', 'instrucciones', 'guía'],
            'qué': ['qué', 'cuál', 'definición', 'significado', 'concepto'],
            'por qué': ['por qué', 'razón', 'causa', 'motivo', 'explicación'],
            'error': ['error', 'problema', 'bug', 'fallo', 'solución', 'arreglar'],
            'receta': ['receta', 'cocinar', 'ingredientes', 'preparación', 'cocina'],
            'ejemplo': ['ejemplo', 'ejemplar', 'muestra', 'modelo', 'caso'],
        }
        
        # Identificar términos clave en la query
        expanded_terms = set(query.split())
        for term in query.split():
            if term in query_expansions:
                expanded_terms.update(query_expansions[term])
        
        # Añadir términos relacionados semánticamente
        semantic_terms = self._find_semantic_terms(query, list(expanded_terms))
        expanded_terms.update(semantic_terms)
        
        return ' '.join(list(expanded_terms)[:10])  # Limitar a 10 términos
    
    def _find_semantic_terms(self, query: str, existing_terms: List[str]) -> List[str]:
        """Encuentra términos semánticamente relacionados usando embeddings"""
        try:
            # Generar embedding de la query
            query_embedding = self._generate_embedding(query)
            
            # Buscar términos similares en el espacio de embeddings
            similar_terms = []
            for term in existing_terms:
                if len(term) > 3:  # Solo para términos significativos
                    term_embedding = self._generate_embedding(term)
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [term_embedding]
                    )[0][0]
                    
                    if similarity > 0.6:
                        similar_terms.append(term)
            
            return similar_terms
        except:
            return []
    
    def _filter_and_rerank_results(self, results: Dict, original_query: str, 
                                  enhanced_query: str, threshold: float, 
                                  use_reranking: bool) -> Dict:
        """Filtra y rerankea resultados basado en relevancia semántica mejorada"""
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []
        relevance_scores = []
        
        query_terms = set(original_query.lower().split())
        enhanced_terms = set(enhanced_query.lower().split())
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0] if results['distances'] else [1.0] * len(results['documents'][0])
        )):
            # Calcular similitud
            similarity = 1 - distance
            
            # Calcular relevancia semántica mejorada
            relevance_score = self._calculate_enhanced_relevance_score(
                doc, metadata, query_terms, enhanced_terms, original_query
            )
            
            # Puntuación combinada (60% similitud, 40% relevancia semántica)
            combined_score = (similarity * 0.6) + (relevance_score * 0.4)
            
            if combined_score >= threshold:
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
                relevance_scores.append(combined_score)
        
        # Re-ranking basado en puntuación combinada
        if use_reranking and filtered_docs:
            sorted_indices = np.argsort(relevance_scores)[::-1]  # Orden descendente
            
            filtered_docs = [filtered_docs[i] for i in sorted_indices]
            filtered_metadatas = [filtered_metadatas[i] for i in sorted_indices]
            filtered_distances = [filtered_distances[i] for i in sorted_indices]
        
        return {
            'documents': filtered_docs,
            'metadatas': filtered_metadatas,
            'distances': filtered_distances
        }
    
    def _calculate_enhanced_relevance_score(self, doc_text: str, metadata: Dict, 
                                          query_terms: set, enhanced_terms: set, 
                                          original_query: str) -> float:
        """Calcula puntuación de relevancia semántica mejorada"""
        doc_text_lower = doc_text.lower()
        score = 0
        
        # 1. Coincidencia en nombre de archivo (alta prioridad)
        metadata_name = metadata.get('name', '')
        if metadata_name:
            metadata_name = metadata_name.lower()
            for term in query_terms:
                if term in metadata_name:
                    score += 3
        
        # 2. Coincidencia en keywords
        keywords_str = metadata.get('keywords', '')
        if keywords_str:
            keywords_list = [k.strip().lower() for k in keywords_str.split(',')]
            for term in query_terms:
                if term in keywords_list:
                    score += 2
        
        # 3. Coincidencia en categoría
        query_category = self._detect_semantic_category(original_query, list(query_terms))
        doc_category = metadata.get('category', 'general')
        if query_category == doc_category:
            score += 2
        
        # 4. Coincidencia en contenido (términos mejorados)
        content_matches = sum(1 for term in enhanced_terms if term in doc_text_lower)
        score += content_matches * 0.5
        
        # 5. Coincidencia exacta de frases
        for term in query_terms:
            if len(term) > 3 and f" {term} " in f" {doc_text_lower} ":
                score += 1
        
        # Normalizar score
        max_possible_score = 15  # Ajustado según los nuevos factores
        return min(score / max_possible_score, 1.0)
    
    def list_collections(self):
        """Lista todas las colecciones con metadata"""
        try:
            collections = self.client.list_collections()
            collections_info = []
            
            metadata_path = os.path.join(self.data_path, "collections_metadata.json")
            metadata = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.error(f"Error cargando metadata: {e}")
                    metadata = {}
            
            for collection in collections:
                collection_metadata = metadata.get(collection.name, {})
                collection_info = {
                    'name': collection.name,
                    'metadata': collection.metadata,
                    'count': collection_metadata.get('document_count', 0),
                    'chunk_count': collection_metadata.get('chunk_count', 0),
                    'created_at': collection_metadata.get('created_at', ''),
                    'updated_at': collection_metadata.get('updated_at', '')
                }
                collections_info.append(collection_info)
            
            return collections_info
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str):
        """Elimina una colección"""
        try:
            self.client.delete_collection(collection_name)
            
            # Eliminar de metadata
            metadata_path = os.path.join(self.data_path, "collections_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    metadata.pop(collection_name, None)
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Error eliminando metadata: {e}")
            
            logger.info(f"Colección '{collection_name}' eliminada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str):
        """Obtiene estadísticas detalladas de una colección"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            # Calcular estadísticas adicionales
            metadata_path = os.path.join(self.data_path, "collections_metadata.json")
            additional_stats = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    additional_stats = metadata.get(collection_name, {})
                except Exception as e:
                    logger.error(f"Error cargando estadísticas: {e}")
            
            return {
                'document_count': count,
                'chunk_count': additional_stats.get('chunk_count', 0),
                'created_at': additional_stats.get('created_at', ''),
                'updated_at': additional_stats.get('updated_at', ''),
                'embedding_model': collection.metadata.get('embedding_model', ''),
                'embedding_size': collection.metadata.get('embedding_size', '')
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas para {collection_name}: {e}")
            return {'document_count': 0, 'chunk_count': 0}
    
    def get_collection_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """Obtiene la lista de documentos únicos en una colección"""
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.get()
            
            # Agrupar por file_path para obtener documentos únicos
            unique_docs = {}
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    file_path = metadata.get('file_path')
                    if file_path and file_path not in unique_docs:
                        # Crear entrada de documento único
                        unique_docs[file_path] = {
                            'name': metadata.get('name', 'Unknown'),
                            'type': metadata.get('type', 'unknown'),
                            'file_path': file_path,
                            'size': metadata.get('size', 'N/A'),
                            'date': metadata.get('date', 'N/A'),
                            'chunk_count': 1,  # Inicializar contador de chunks
                            'category': metadata.get('category', 'general'),
                            'language': metadata.get('language', ''),
                            'keywords': metadata.get('keywords', '')
                        }
                    elif file_path in unique_docs:
                        # Incrementar contador de chunks para este documento
                        unique_docs[file_path]['chunk_count'] += 1
            
            return list(unique_docs.values())
        except Exception as e:
            logger.error(f"Error getting documents from {collection_name}: {e}")
            return []
    
    def remove_document(self, collection_name: str, file_path: str) -> bool:
        """Elimina todos los chunks de un documento específico"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Buscar todos los IDs asociados con este file_path
            results = collection.get(
                where={"file_path": file_path},
                include=["metadatas", "documents"]
            )
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                # Actualizar la metadata de la colección
                self._update_collection_metadata_after_deletion(collection_name, len(results['ids']))
                logger.info(f"Documento {file_path} eliminado de {collection_name}")
                return True
            else:
                logger.warning(f"Documento {file_path} no encontrado en {collection_name}")
                return False
        except Exception as e:
            logger.error(f"Error removing document {file_path} from {collection_name}: {e}")
            return False
    
    def _update_collection_metadata_after_deletion(self, collection_name: str, deleted_count: int):
        """Actualiza la metadata de la colección después de eliminar documentos"""
        metadata_path = os.path.join(self.data_path, "collections_metadata.json")
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error cargando metadata: {e}")
                metadata = {}
        
        if collection_name in metadata:
            metadata[collection_name]['document_count'] = max(0, metadata[collection_name].get('document_count', 0) - deleted_count)
            metadata[collection_name]['chunk_count'] = max(0, metadata[collection_name].get('chunk_count', 0) - deleted_count)
            metadata[collection_name]['updated_at'] = datetime.now().isoformat()
            
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error updating collection metadata: {e}")

# Singleton para gestión global de ChromaManager
_chroma_manager_instance = None

def get_chroma_manager(data_path="data/chromadb", embedding_model_name=None, device_override=None):
    """Obtiene la instancia singleton de ChromaManager con configuración personalizada"""
    global _chroma_manager_instance
    if _chroma_manager_instance is None:
        _chroma_manager_instance = ChromaManager(
            data_path=data_path,
            embedding_model_name=embedding_model_name,
            device_override=device_override
        )
    return _chroma_manager_instance