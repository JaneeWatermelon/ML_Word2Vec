import numpy as np
import torch
from numba import jit

from core.base import sigmoid, softmax


def grad_descent_normal(
        v_word: str, 
        u_word: str, 
        v_voc: np.ndarray, 
        u_voc: np.ndarray, 
        vocabulary: dict, 
        words: list[str], 
        alpha: float, 
    ):
    v_i = vocabulary[v_word]
    u_i = vocabulary[u_word]
    v_c = v_voc[v_i]
    u_c = u_voc[u_i]

    dp_u_all = np.dot(u_voc, v_c)
    dot_prod_softmax = softmax(dp_u_all)

    new_v_c = v_c - alpha*(-u_c + (dot_prod_softmax.reshape(dot_prod_softmax.shape[0], 1)*u_voc).sum())
    v_voc[v_i] = new_v_c

    u_voc = u_voc - alpha*(np.multiply.outer(dot_prod_softmax, v_c))
    u_voc[u_i] = u_c - alpha*v_c*(dot_prod_softmax[u_i] - 1)

    return v_voc, u_voc

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

    # possible_samples = set(range(0, len(vocabulary)-1)) - {u_i, v_i}
    # if u_i in possible_samples:
    #     possible_samples.remove(u_i)
    # if v_i in possible_samples:
    #     possible_samples.remove(v_i)
    # k_neg_samples = np.random.choice(list(possible_samples), size=k)

    pos_score = np.dot(u_c, v_c)
    sigma_pos = 1 / (1 + np.exp(-pos_score))

    neg_scores = np.dot(u_voc[k_neg_samples], v_c)
    sigma_neg = 1 / (1 + np.exp(-neg_scores))

    loss_v_c = (sigma_pos - 1) * u_c + (sigma_neg[:, np.newaxis] * u_voc[k_neg_samples]).sum(axis=0)
    loss_u_c = (sigma_pos - 1) * v_c
    loss_u_k = np.outer(sigma_neg, v_c)

    # dist_koef = kwargs["dist_koef"]
    # print(dist_koef)

    v_voc[v_i] -= alpha*dist_koef*loss_v_c
    u_voc[u_i] -= alpha*dist_koef*loss_u_c
    u_voc[k_neg_samples] -= alpha*dist_koef*loss_u_k

    epsilon = 1e-10

    curr_loss = -np.log(sigma_pos + epsilon) - np.log(1 - sigma_neg + epsilon).sum(axis=0)

    # print(kwargs.get("epoch"))
    # if kwargs.get("epoch") == 5:
    #     sigma_negs = sigmoid(np.dot(u_voc[k_neg_samples], v_c))
    #     sigma_pos = sigmoid(np.dot(u_c, v_c))
    #     print(f"sigma_pos: {sigma_pos:.4f}, sigma_negs_avg: {np.mean(sigma_negs):.4f}, grad_norm: {np.linalg.norm(loss_v_c):.4f}")

    return curr_loss


def pytorch_grad_descent(
        v_word, 
        u_word, 
        v_voc, 
        u_voc, 
        vocabulary, 
        k=5, 
        alpha=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Переносим данные на GPU
    v_voc_gpu = torch.tensor(v_voc, device=device)
    u_voc_gpu = torch.tensor(u_voc, device=device)
    
    v_i = vocabulary[v_word]
    u_i = vocabulary[u_word]
    
    # Вектора на GPU
    v_c = v_voc_gpu[v_i]
    u_c = u_voc_gpu[u_i]
    
    # # Выбираем негативные индексы
    # neg_indices = []
    # for _ in range(k):
    #     while True:
    #         idx = np.random.randint(0, len(vocabulary)-1)
    #         if idx != u_i and idx != v_i and idx not in neg_indices:
    #             neg_indices.append(idx)
    #             break

    possible_samples = set(range(0, len(vocabulary)-1))
    if u_i in possible_samples:
        possible_samples.remove(u_i)
    if v_i in possible_samples:
        possible_samples.remove(v_i)
    # k_neg_samples = np.random.randint(0, len(vocabulary)-1, k)
    neg_indices = np.random.choice(list(possible_samples), size=k)
    
    u_negs = u_voc_gpu[neg_indices]
    
    # Вычисляем на GPU
    pos_score = torch.dot(v_c, u_c)
    sigma_pos = torch.sigmoid(pos_score)
    
    neg_scores = torch.mv(u_negs, v_c)
    sigma_negs = torch.sigmoid(neg_scores)
    
    # Градиенты
    grad_v = (sigma_pos - 1) * u_c + torch.mv(u_negs.T, sigma_negs)
    grad_u_c = (sigma_pos - 1) * v_c
    grad_u_negs = torch.outer(sigma_negs, v_c)
    
    # Обновление
    v_voc_gpu[v_i] -= alpha * grad_v
    u_voc_gpu[u_i] -= alpha * grad_u_c
    u_voc_gpu[neg_indices] -= alpha * grad_u_negs

    curr_loss = (-torch.log(sigma_pos) - torch.log(1 - sigma_negs).sum()).item()
    
    # Возвращаем на CPU
    return v_voc_gpu.cpu().numpy(), u_voc_gpu.cpu().numpy(), curr_loss