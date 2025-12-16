from collections import Counter, defaultdict
import os
import re
import numpy as np
from nltk import WordPunctTokenizer

from core.vars import ASSETS_ROOT, UNK_VAL

def softmax(vector: np.array) -> np.array:
    return np.exp(vector) / np.exp(vector).sum()

def sigmoid(vector: np.array) -> np.array:
    return 1 / (1 + np.exp(-vector))

def text_preprocessor(texts: list[list[str]], min_freq: int=5) -> tuple[list[list[str]], dict[str, int]]:
    tokenizer = WordPunctTokenizer()

    tokens_counts = Counter()
    total_tokens = 0
    vocabulary = {}

    tokens = []

    for i in range(len(texts)):
        text = texts[i]
        # text = re.sub(r"\W", " ", text)
        # text = re.sub(r"\s+", " ", text)
        # texts[i] = text

        text = re.sub(r'[.,!?;:"\'()\[\]{}<>«»„“”\-–—/\\|@#$%^&*_+=~`]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)  # Множественные пробелы/табы → один пробел
        text = re.sub(r'\n[ \t]+\n', '\n\n', text)  # Убираем пробелы между переносами
        texts[i] = text

    

    for i in range(len(texts)):
        rows = texts[i].split("\n")
        for row in rows:
            if len(row.strip()) == 0:
                continue
            words = tokenizer.tokenize(row.lower())
            tokens_counts.update(words)
            total_tokens += len(words)
            tokens.append(words)

    idx = 0
    for word, count in tokens_counts.most_common():
        if count >= min_freq and word not in vocabulary:
            vocabulary[word] = idx
            idx += 1
        else:
            break

    vocabulary[UNK_VAL] = idx

    return tokens, vocabulary

def make_vocabulary(texts_map: dict[str, list[str]], min_appears: int=1, is_save: bool=False, save_dir: str=ASSETS_ROOT, file_name: str="voc.txt"):
    voc = set()
    words_count_map = defaultdict(int)
    for key, text_list in texts_map.items():
        for word in text_list:
            words_count_map[word] += 1
            voc.add(word)
        # voc |= set(text.split())

    unk_words = set(UNK_VAL)

    for word in voc:
        if words_count_map[word] <= min_appears:
            unk_words.add(word)

    voc -= unk_words

    voc = list(voc)
    voc.append(UNK_VAL)

    vocabulary = dict(zip(voc, range(len(voc))))
    print(len(vocabulary))

    if is_save:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, file_name), "w", encoding="UTF-8") as f:
            for word in voc:
                f.write(word + "\n")

    return voc, vocabulary, unk_words

def load_vocabulary(save_dir: str=ASSETS_ROOT, file_name: str="voc.txt"):
    with open(os.path.join(save_dir, file_name), "r", encoding="UTF-8") as f:
        voc = f.readlines()

    vocabulary = dict(zip(voc, range(len(voc))))
    unk_words = []
                
    return voc, vocabulary, unk_words

def create_training_batches(words, vocabulary, window_size=2, batch_size=32):
    """Создание батчей для обучения"""
    n = len(words)
    
    target_batch = []
    context_batch = []
    
    for i in range(n):
        central_word = words[i]
        if central_word not in vocabulary:
            central_word = UNK_VAL
        
        # Контекстные слова
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        
        for j in range(start, end):
            if j == i:
                continue
                
            context_word = words[j]
            if context_word not in vocabulary:
                context_word = UNK_VAL
            
            target_batch.append(vocabulary[central_word])
            context_batch.append(vocabulary[context_word])
            
            if len(target_batch) == batch_size:
                yield np.array(target_batch), np.array(context_batch)
                target_batch = []
                context_batch = []
    
    if target_batch:
        yield np.array(target_batch), np.array(context_batch)
