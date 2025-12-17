from collections import Counter
import re
from numba import jit, prange
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk import WordPunctTokenizer
from core.loader import load_texts_from_directory
from core.vars import TEXT_ROOT, ASSETS_ROOT, UNK_VAL

@jit(nopython=True)
def grad_descent_neg_sampling(
        v_i: int, 
        u_i: int, 
        v_voc: np.ndarray, 
        u_voc: np.ndarray, 
        alpha: float, 
        k_neg_samples: np.array,
        dist_koef: float,
    ):
    v_c = v_voc[v_i]
    u_c = u_voc[u_i]

    pos_score = np.dot(u_c, v_c)
    sigma_pos = 1 / (1 + np.exp(-pos_score))

    neg_scores = np.dot(u_voc[k_neg_samples], v_c)
    sigma_neg = 1 / (1 + np.exp(-neg_scores))

    loss_v_c = (sigma_pos - 1) * u_c + (sigma_neg[:, np.newaxis] * u_voc[k_neg_samples]).sum(axis=0)
    loss_u_c = (sigma_pos - 1) * v_c
    loss_u_k = np.outer(sigma_neg, v_c)

    v_voc[v_i] -= alpha*dist_koef*loss_v_c
    u_voc[u_i] -= alpha*dist_koef*loss_u_c
    u_voc[k_neg_samples] -= alpha*dist_koef*loss_u_k

    epsilon = 1e-10

    curr_loss = -np.log(sigma_pos + epsilon) - np.log(1 - sigma_neg + epsilon).sum(axis=0)

    return curr_loss

def text_preprocessor(texts: list[list[str]], min_freq: int=5) -> tuple[list[list[str]], dict[str, int]]:
    tokenizer = WordPunctTokenizer()

    tokens_counts = Counter()
    total_tokens = 0
    vocabulary = {}

    tokens = []

    for i in range(len(texts)):
        text = texts[i]

        text = re.sub(r'[.,!?;:"\'()\[\]{}<>«»„“”\-–—/\\|@#$%^&*_+=~`]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)  # Множественные пробелы/табы -> один пробел
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

# JIT-совместимая версия основной логики
@jit(nopython=True, parallel=True)
def train_epochs_jit(
        v_voc, 
        u_voc, 
        tokens_indices, 
        vocabulary_size, 
        window_size, 
        k, 
        n_epochs, 
        initial_alpha
    ):
    """
    JIT-совместимая функция обучения word2vec
    """
    n_sentences = len(tokens_indices)
    dimen = v_voc.shape[1]
    alpha = initial_alpha
    
    total_losses = np.zeros(n_epochs)
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        total_words = 0
        
        # Проходим по всем предложениям
        for t_i in prange(n_sentences):
            words = tokens_indices[t_i]
            if len(words) == 0:
                continue
                
            n = min(len(words), 500000)  # Ограничение длины
            
            k_neg_samples_vec = np.random.randint(0, vocabulary_size, 
                                                 size=(n, 2*window_size, k))
            
            # Коэффициенты расстояний
            first_part_dist = np.arange(window_size, 0, -1)
            dist_koefs = np.concatenate((first_part_dist, first_part_dist[::-1]))
            dist_koefs = 1.0 / dist_koefs
            
            # Для каждого целевого слова
            for i in range(n):
                # Определяем контекстные слова
                start_idx = max(i - window_size, 0)
                end_idx = min(i + window_size + 1, n)
                
                # Собираем индексы контекстных слов
                context_indices = []
                for j in range(start_idx, end_idx):
                    if j != i:
                        context_indices.append(words[j])
                
                v_i = words[i]
                
                # Обновляем для каждого контекстного слова
                for cont_i, u_i in enumerate(context_indices):
                    # if cont_i < 2 * window_size:  # Проверка границ
                    # Получаем коэффициент расстояния
                    dist_idx = min(cont_i, len(dist_koefs) - 1)
                    dist_koef = dist_koefs[dist_idx]
                    
                    # Получаем негативные сэмплы
                    neg_samples = k_neg_samples_vec[i, cont_i]
                    
                    # Обновляем вектора
                    curr_loss = grad_descent_neg_sampling(
                        v_i=v_i,
                        u_i=u_i,
                        v_voc=v_voc,
                        u_voc=u_voc,
                        alpha=alpha,
                        k_neg_samples=neg_samples,
                        dist_koef=dist_koef
                    )
                    
                    epoch_loss += curr_loss
                    total_words += 1
        
        # Усредняем лосс
        if total_words > 0:
            avg_loss = epoch_loss / total_words
            total_losses[epoch] = avg_loss
        
        # Уменьшаем скорость обучения
        alpha *= 0.99
    
    return v_voc, u_voc, total_losses

def main(
        dimen: int = 100,
        alpha: float = 0.05,
        window_size: int = 2,
        k: int = 3,
        n_epochs: int = 1,
    ):
    # Загрузка и предобработка текстов
    print("Загрузка текстов...")
    texts_map = load_texts_from_directory("quora_text", n_max=10)
    tokens, vocabulary = text_preprocessor(list(texts_map.values()))
    
    voc = list(vocabulary.keys())
    vocabulary_size = len(voc)
    print(f"Размер словаря: {vocabulary_size}")
    
    # Преобразуем токены в индексы
    tokens_indices = []
    for sentence in tokens:
        indices = []
        for word in sentence[:500000]:  # Ограничение длины
            idx = vocabulary.get(word, vocabulary.get(UNK_VAL, 0))
            indices.append(idx)
        tokens_indices.append(np.array(indices, dtype=np.int32))
    
    # Инициализируем вектора
    v_voc = np.random.randn(vocabulary_size, dimen).astype(np.float32) * 0.1
    u_voc = np.random.randn(vocabulary_size, dimen).astype(np.float32) * 0.1
    
    print(f"Начало обучения: dimen={dimen}, alpha={alpha}, epochs={n_epochs}")
    
    # Запускаем JIT-совместимое обучение
    start_time = time.time()
    v_voc, u_voc, losses = train_epochs_jit(
        v_voc, u_voc, 
        tokens_indices, 
        vocabulary_size,
        window_size, 
        k, 
        n_epochs, 
        alpha
    )
    training_time = time.time() - start_time
    
    print(f"Обучение завершено за {training_time:.2f} секунд")
    
    # Сохраняем результаты
    np.savez(os.path.join(ASSETS_ROOT, "v_u_vocs.npz"), v_voc=v_voc, u_voc=u_voc)
    with open(os.path.join(ASSETS_ROOT, "voc.txt"), "w", encoding="utf-8") as f:
        for word in voc:
            f.write(word + "\n")
    print("Вектора сохранены")
    
    # Выводим статистику по эпохам
    for epoch, loss in enumerate(losses):
        if loss > 0:
            print(f"Epoch {epoch+1}: loss = {loss:.6f}")
    
    print("\nАнализ семантической близости слов...")
    analyze_embeddings(v_voc, voc)

def show_scatter_plot(emb_2d, voc_2d):
    plt.figure(figsize=(12, 10))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5)

    for i, word in enumerate(voc_2d):
        plt.annotate(word, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=12)

    plt.show()

def analyze_embeddings(embeddings, voc, top_n=10, show_per: int=250):
    """
    Анализ эмбеддингов (не JIT-часть)
    """
    n_words = len(voc)

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    words_left_to_print = 20
    
    for i in range(n_words):
        vec = embeddings[i]
        
        # Нормируем вектора для косинусного сходства
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms_vec = np.linalg.norm(vec)
        
        # Косинусное сходство
        cos_sim = np.dot(embeddings, vec) / (norms.flatten() * norms_vec + 1e-8)
        
        # Находим наиболее похожие слова
        similar_indices = np.argsort(-cos_sim)[1:top_n+1]  # исключаем само слово
        
        # Проверяем порог сходства
        if cos_sim[similar_indices[0]] >= 0.7:
            print(f"\nБлижайшие слова для: '{voc[i]}'")
            for j, idx in enumerate(similar_indices[:5]):
                word = voc[idx]
                similarity = cos_sim[idx]
                print(f"  {j+1}. {word}: {similarity:.3f}")
            # input()

            words_left_to_print -= 1

            # show_scatter_plot(emb_2d=emb_2d[similar_indices], voc_2d=np.array(voc)[similar_indices])

        if words_left_to_print <= 0:
            break

if __name__ == "__main__":
    if os.name == 'nt':
        os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
        os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

    # main(
    #     dimen=100,
    #     alpha=0.05,
    #     window_size=2,
    #     k=3,
    #     n_epochs=5
    # )

    npz_files = np.load(os.path.join(ASSETS_ROOT, "v_u_vocs.npz"))
    embeddings = npz_files["v_voc"]
    # print(embeddings["v_voc"])
    voc = []
    with open(os.path.join(ASSETS_ROOT, "voc.txt"), "r", encoding="utf-8") as f:
        voc = f.readlines()
    
    voc = list(map(lambda x: x.strip(), voc))

    analyze_embeddings(embeddings, voc=voc)