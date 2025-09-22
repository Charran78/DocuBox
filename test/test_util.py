import sys
import pathlib

# Añadir el directorio actual al path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from utils import process_files, extract_cover, extract_text_from_file, detect_language

print("Testing utils functions...")
print("process_files:", callable(process_files))
print("extract_cover:", callable(extract_cover))
print("extract_text_from_file:", callable(extract_text_from_file))
print("detect_language:", callable(detect_language))

# Test detect_language
test_text = "This is a test in English"
print(f"Language detection for '{test_text}':", detect_language(test_text))

test_text_es = "Esto es una prueba en español"
print(f"Language detection for '{test_text_es}':", detect_language(test_text_es))