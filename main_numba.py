from numba import jit, prange
import os
import time
import numpy as np
import pandas as pd
from core.base import text_preprocessor
from core.grad_descent import grad_descent_neg_sampling
from core.loader import load_texts_from_directory
from core.vars import TEXT_ROOT, ASSETS_ROOT, UNK_VAL

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
    # Настройка окружения (только для Windows)
    if os.name == 'nt':
        os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
        os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
    
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
    print("Вектора сохранены")
    
    # Выводим статистику по эпохам
    for epoch, loss in enumerate(losses):
        if loss > 0:
            print(f"Epoch {epoch+1}: loss = {loss:.6f}")
    
    # Анализ результатов (не JIT-часть)
    print("\nАнализ семантической близости слов...")
    analyze_embeddings(v_voc, voc)

def analyze_embeddings(embeddings, vocabulary, top_n=10):
    """
    Анализ эмбеддингов (не JIT-часть)
    """
    n_words = len(vocabulary)
    
    for i in range(n_words):  # Проверяем только первые 20 слов для демонстрации
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
            print(f"\nБлижайшие слова для: '{vocabulary[i]}'")
            for j, idx in enumerate(similar_indices[:5]):
                word = vocabulary[idx]
                similarity = cos_sim[idx]
                print(f"  {j+1}. {word}: {similarity:.3f}")
            input()

if __name__ == "__main__":
    main(
        dimen=100,
        alpha=0.05,
        window_size=2,
        k=3,
        n_epochs=5
    )