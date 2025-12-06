import os
from core.vars import TEXT_ROOT

def load_text_from_file(file_path, encoding='utf-8') -> str:
    """
    Загружает текст из одного файла.
    
    Args:
        file_path: путь к файлу
        encoding: кодировка файла (по умолчанию 'utf-8')
    
    Returns:
        str: содержимое файла
        
    Raises:
        FileNotFoundError: если файл не найден
        UnicodeDecodeError: если возникли проблемы с кодировкой
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            text = file.read()
        return text.strip()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        raise
    except UnicodeDecodeError:
        raise

def load_texts_from_directory(
        directory_path, 
        extensions=None, 
        encoding='utf-8',
        recursive=False
    ):
    """
    Загружает текст из нескольких файлов в директории.
    
    Args:
        directory_path: путь к директории с файлами
        extensions: список расширений файлов для обработки 
                   (например, ['.txt', '.json'])
                   Если None, обрабатываются все файлы
        encoding: кодировка файлов
        recursive: если True, ищет файлы во всех поддиректориях
    
    Returns:
        dict: словарь {имя_файла: текст}
    """
    if extensions is None:
        extensions = ['.txt']
    
    texts = {}
    
    if recursive:
        # Рекурсивный обход всех поддиректорий
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                if any(filename.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, filename)
                    try:
                        text = load_text_from_file(file_path, encoding)
                        texts[file_path] = text
                    except Exception as e:
                        print(f"Ошибка при чтении файла {file_path}: {e}")
    else:
        # Только файлы в указанной директории
        try:
            files = os.listdir(directory_path)
        except FileNotFoundError:
            print(f"Ошибка: Директория '{directory_path}' не найдена.")
            return texts
            
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in extensions):
                try:
                    text = load_text_from_file(file_path, encoding)
                    texts[filename] = text
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
    
    print(f"Загружено {len(texts)} файлов")
    return texts