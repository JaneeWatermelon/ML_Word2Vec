# from numba import jit, cuda
import math
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

def main(
        dimen: int=100,
        alpha: float=0.05,
        window_size: int=2,
        batch_size: int=10**5,
        k: int=3,
        n_epoches: int=1,
    ):
    os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

    texts_map = load_texts_from_directory("texts", n_max=10)

    tokens, vocabulary = text_preprocessor(list(texts_map.values()))
    voc = list(vocabulary.keys())
    print(len(vocabulary))
    print(len(tokens))

    v_voc = np.random.randn(len(vocabulary), dimen) * 0.1
    u_voc = np.random.randn(len(vocabulary), dimen) * 0.1

    for epoch in range(1, n_epoches+1):
        print(f"Started epoch {epoch} with alpha = {alpha}:")
        epoch_loss = 0
        total_words = 0
        for words in tokens:
            n = len(words)
            print(n)
            start_t = time.time()

            k_neg_samples_vec = np.random.choice(list(range(len(vocabulary))), size=(n, 2*window_size, k))

            first_part_dist = list(range(window_size, 0, -1))
            dist_koefs = first_part_dist + first_part_dist[::-1]

            dist_koefs = 1 / np.array(dist_koefs)
            dist_koefs = np.outer(np.ones(n), dist_koefs)

            for m in range(0, min(window_size, n)):
                dist_koefs[m] = np.roll(dist_koefs[m], shift=(m-window_size))

            total_processed_words = 0

            batch_i = 0
            total_batches = math.ceil(n / batch_size)

            for target_indices, context_indices in create_training_batches(
                words, 
                vocabulary=vocabulary,
                window_size=window_size,
                batch_size=batch_size,
            ):
                batch_i += 1
                batch_start_t = time.time()
                batch_len_actual = len(target_indices)

                # v_batch = v_voc[target_indices]  # [batch, dim]
                # u_batch = u_voc[context_indices]  # [batch, dim]
                # print(v_batch)
                # print(v_batch[0])


                for i in range(batch_len_actual):
                    first_i = total_processed_words + i // (2*window_size)
                    second_i = i % (2*window_size)

                    v_voc, u_voc, curr_loss = grad_descent_neg_sampling(
                        v_i=target_indices[i],
                        u_i=context_indices[i],
                        v_voc=v_voc,
                        u_voc=u_voc,
                        alpha=alpha,
                        k_neg_samples=k_neg_samples_vec[first_i, second_i],
                        dist_koef=dist_koefs[first_i, second_i],
                        # epoch=epoch
                    )
                    epoch_loss += curr_loss

                # print(f"Before min: {total_processed_words}")
                # print(f"Before max: {total_processed_words + batch_len_actual // (2*window_size) - 1}")

                total_processed_words += batch_len_actual // (2*window_size) + batch_len_actual % (2*window_size)
                # print(f"After: {total_processed_words}")

                print(f"Time for batch {batch_i}/{total_batches}: {(time.time() - batch_start_t):.3f} seconds")

            total_words += n

            print(f"Processed {len(words)} words in {(time.time() - start_t):.3f} seconds")

        print(f"epoch_loss: {epoch_loss / total_words:.6f}")
        alpha *= 0.99

    np.savez(os.path.join(ASSETS_ROOT, "v_u_vocs.npz"), v_voc, u_voc)
    
    for i, vec in enumerate(v_voc):
        norms = np.linalg.norm(v_voc, axis=1) * np.linalg.norm(vec)
        cosin_simularity = np.dot(v_voc, vec) / norms
        # print(cosin_simularity)

        simularity_df = pd.concat([pd.Series(voc, name="word"), pd.Series(cosin_simularity, name="simularity")], axis=1)
        simularity_df = simularity_df.sort_values(by="simularity", ascending=False).iloc[1:]
        # v_voc_df = pd.concat([pd.Series(voc, name="word"), pd.Series(vec, name="embedding")], axis=1)

        if simularity_df.iloc[0]["simularity"] >= 0.5:
            print(f"Близайшие контексты для слова: {voc[i]}")
            # print(simularity_df.sort_values(by="simularity", ascending=False).head(5)) 
            print(simularity_df.head(5)) 
            input()   

if __name__ == "__main__":
    main(
        n_epoches=1
    )

    
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