import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.loader import load_texts_from_directory
from core.vars import TEXT_ROOT

def softmax(vector: np.array):
    return np.exp(vector) / np.exp(vector).sum()

if __name__ == "__main__":
    os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

    texts_map = load_texts_from_directory(TEXT_ROOT)
    for key, text in texts_map.items():
        text = re.sub(r"\W", " ", text).lower()
        text = re.sub(r"\s+", " ", text).lower()
        texts_map[key] = text

    voc = list(set(" ".join(list(texts_map.values())).split()))
    vocabulary = dict(zip(voc, range(len(voc))))
    print(vocabulary)
    vocabulary["UNK"] = -1

    dimen = 300
    alpha = 0.01


    v_vec = np.random.randn(len(voc), dimen) * 0.1
    u_vec = np.random.randn(len(voc), dimen) * 0.1

    for key, text in texts_map.items():
        words = text.strip().split()
        n = len(words)
        j = 2
        k = 15
        for i in range(len(words)):
            if words[i] in vocabulary:
                context_words = words[max(i - j, 0):i] + words[i+1:min(i + j + 1, n)]
                central_vec = v_vec[vocabulary[words[i]]]

                for cont_word in context_words:
                    dot_prod_u_all = np.dot(u_vec, central_vec)
                    cont_i = vocabulary[cont_word]
                    curr_context_vec = u_vec[cont_i]
                    dot_prod_softmax = softmax(dot_prod_u_all)
                    k_neg_samples = np.random.randint(0, len(voc), k)

                    new_central_vec = central_vec - alpha*(-curr_context_vec + (dot_prod_softmax.reshape(dot_prod_softmax.shape[0], 1)*u_vec).sum())
                    v_vec[vocabulary[words[i]]] = new_central_vec

                    u_vec[k_neg_samples] = u_vec[k_neg_samples] - alpha*(np.multiply.outer(dot_prod_softmax[k_neg_samples], central_vec))
                    u_vec[cont_i] = curr_context_vec - alpha*central_vec*(dot_prod_softmax[cont_i] - 1)
                    # u_vec = new_u_vec.copy()
            else:
                pass
    
    # for central_vec in v_vec:
    #     plot = sns.barplot(
    #         central_vec
    #     )
    #     plt.show()
    
    for i, vec in enumerate(v_vec):
        cosin_simularity = np.dot(v_vec, vec)

        simularity_df = pd.concat([pd.Series(voc, name="word"), pd.Series(cosin_simularity, name="simularity")], axis=1)
        simularity_df = simularity_df.sort_values(by="simularity", ascending=False).iloc[1:]
        # v_vec_df = pd.concat([pd.Series(voc, name="word"), pd.Series(vec, name="embedding")], axis=1)

        if simularity_df.iloc[0]["simularity"] >= 0.9:
            print(f"Близайшие контексты для слова: {voc[i]}")
            # print(simularity_df.sort_values(by="simularity", ascending=False).head(5)) 
            print(simularity_df.head(5)) 
            input()   

    
