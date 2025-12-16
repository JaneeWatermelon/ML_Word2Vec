from numba import jit
from http import client
import os
import re
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.base import create_training_batches, text_preprocessor
from core.grad_descent import pytorch_grad_descent, grad_descent_neg_sampling
from core.loader import load_texts_from_directory
from core.vars import TEXT_ROOT, ASSETS_ROOT, UNK_VAL
from multiprocessing import Process

# @jit(nopython=True)
def main(
        dimen: int=100,
        alpha: float=0.05,
        window_size: int=2,
        k: int=3,
        n_epoches: int=1,
    ):
    os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

    texts_map = load_texts_from_directory("quora_text", n_max=10)

    tokens, vocabulary = text_preprocessor(list(texts_map.values()))
    voc = list(vocabulary.keys())
    print(len(vocabulary))

    v_voc = np.random.randn(len(vocabulary), dimen) * 0.1
    u_voc = np.random.randn(len(vocabulary), dimen) * 0.1

    for epoch in range(1, n_epoches+1):
        print(f"Started epoch {epoch} with alpha = {alpha}:")
        epoch_loss = 0
        total_words = 0
        next_t_i_print = len(tokens) // 100
        t_i_time = time.time()
        for t_i, words in enumerate(tokens):
            if t_i == next_t_i_print:
                print(f"Time for {t_i}/{len(tokens)} rows: {(time.time() - t_i_time):.3f}")
            words = words[:5 * 10**5]
            n = len(words)
            # print(f"Words length: {n}")
            start_t = time.time()

            k_neg_samples_vec = np.random.choice(list(range(len(vocabulary))), size=(n, 2*window_size, k))

            first_part_dist = list(range(window_size, 0, -1))
            dist_koefs = first_part_dist + first_part_dist[::-1]

            dist_koefs = 1 / np.array(dist_koefs)
            dist_koefs = np.outer(np.ones(n), dist_koefs)
            for m in range(0, min(window_size, n)):
                dist_koefs[m] = np.roll(dist_koefs[m], shift=(m-window_size))

            for i in range(n):
                context_words = words[max(i - window_size, 0):i] + words[i+1:min(i + window_size + 1, n)]

                v_i = int(vocabulary.get(words[i], vocabulary.get(UNK_VAL, 0)))

                for cont_i, cont_word in enumerate(context_words):
                    u_i = int(vocabulary.get(cont_word, vocabulary.get(UNK_VAL, 0)))

                    v_voc, u_voc, curr_loss = grad_descent_neg_sampling(
                        v_i=v_i,
                        u_i=u_i,
                        v_voc=v_voc,
                        u_voc=u_voc,
                        alpha=alpha,
                        k_neg_samples=k_neg_samples_vec[i, cont_i],
                        dist_koef=dist_koefs[i, cont_i],
                    )
                    epoch_loss += curr_loss

            total_words += n

            # print(f"Processed {len(words)} words in {(time.time() - start_t):.3f} seconds")

        print(f"epoch_loss: {epoch_loss / total_words:.6f} | epoch_words: {total_words}")
        alpha *= 0.99

    np.savez(os.path.join(ASSETS_ROOT, "v_u_vocs.npz"), v_voc, u_voc)
    
    for i, vec in enumerate(v_voc):
        norms = np.linalg.norm(v_voc, axis=1) * np.linalg.norm(vec)
        cosin_simularity = np.dot(v_voc, vec) / norms

        simularity_df = pd.concat([pd.Series(voc, name="word"), pd.Series(cosin_simularity, name="simularity")], axis=1)
        simularity_df = simularity_df.sort_values(by="simularity", ascending=False).iloc[1:]
        # v_voc_df = pd.concat([pd.Series(voc, name="word"), pd.Series(vec, name="embedding")], axis=1)

        if simularity_df.iloc[0]["simularity"] >= 0.7:
            print(f"Близайшие контексты для слова: {voc[i]}")
            # print(simularity_df.sort_values(by="simularity", ascending=False).head(5)) 
            print(simularity_df.head(5)) 
            input()   

if __name__ == "__main__":
    main()
    
# alpha = 0.05
# Started epoch 0:
# epoch_loss: 69.55435162732586
# Started epoch 1:
# epoch_loss: 41.75624357574369
# Started epoch 2:
# epoch_loss: 35.3698286225175
# Started epoch 3:
# epoch_loss: 30.386010566998685
# Started epoch 4:
# epoch_loss: 26.83953540818679