import sys
import pathlib
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from chroma_manager import ChromaManager

def test_chroma():
    chroma = ChromaManager()
    print("🔍 Probando ChromaDB...")
    
    # Listar colecciones
    collections = chroma.list_collections()
    print("Colecciones:", collections)
    
    # Crear una colección de prueba
    test_docs = [{
        'name': 'test.txt',
        'content': 'Este es un libro de prueba sobre programación Python',
        'language': 'es',
        'size': 100
    }]
    
    count = chroma.add_documents('test_collection', test_docs)
    print(f"Documentos añadidos: {count}")
    
    # Buscar
    results = chroma.search('test_collection', 'programación')
    print("Resultados de búsqueda:", results)

if __name__ == "__main__":
    test_chroma()
    