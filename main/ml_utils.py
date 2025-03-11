import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging
import os

group_names = {0: "Premium", 1: "Intermediário", 2: "Econômico"}

def perform_kmeans_clustering(precos):
    """
    K-means: clusteriza  preços dos produtos
    
    Args:
        precos (list): lista de preços
        
    Returns:
        tuple: (clusters, kmeans model) ou (None, None) se ocorrerem erros
    """
    try:
        precos_np = np.array(precos).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(precos_np)
        logging.info("K-Means aplicado com sucesso.")
        return clusters, kmeans
    except Exception as e:
        logging.error(f"Erro no K-Means: {e}")
        return None, None


def create_price_cluster_plot(rows, clusters, save_path="static/agrupamento.png"):
    """
    Cria e salva o gráfico com a marcação dos outlaiers.

    Args:
        rows (list): lista de truplas: (id, title, price).
        clusters (numpy.ndarray): Array de clusters.
        save_path (str): diretório onde o gráfico é salvo.

    Returns:
        str
    """
    try:
        logging.info("Criando gráfico de clusters...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        precos = np.array([row[2] for row in rows])
        x_positions = np.linspace(0, 1, len(precos))

        q1, q3 = np.percentile(precos, [25, 75])
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers = (precos < limite_inferior) | (precos > limite_superior)

        cluster_groups = {i: [] for i in range(3)}
        cluster_x_positions = {i: [] for i in range(3)}

        for idx, (id_, title, price) in enumerate(rows):
            cluster_groups[clusters[idx]].append(price)
            cluster_x_positions[clusters[idx]].append(x_positions[idx])

        plt.figure(figsize=(9, 5))
        colors = ['blue', 'green', 'red']

        for i in range(3):
            if cluster_groups[i]:
                plt.scatter(cluster_x_positions[i], cluster_groups[i], 
                            c=colors[i], s=np.array(cluster_groups[i]) / 10, 
                            alpha=0.7, label=group_names[i])
        
        if np.any(outliers):
            plt.scatter(x_positions[outliers], precos[outliers], 
                        color='red', marker='x', s=80, 
                        label="Outliers")

        plt.xlabel("Distribuição de produtos")
        plt.ylabel("Preço (R$)")
        plt.yscale("log")
        plt.title("Agrupamento de Preços dos Produtos (K-Means) com Outliers")
        plt.legend()

        plt.savefig(save_path)
        plt.close()

        logging.info(f"Gráfico salvo em {save_path}")
        return save_path

    except Exception as e:
        logging.error(f"Erro ao criar gráfico: {e}")
        return None


def organize_clusters(rows, clusters):
    """
    organiza os clusters
    
    Args:
        rows (list): lista de truplas: (id, title, price)
        clusters (numpy.ndarray): Array de clusters
        
    Returns:
        dict: Dicionário com chaves, preços e produtos
    """
    agrupamento = {}
    for i, row in enumerate(rows):
        grupo = int(clusters[i])
        grupo_nome = group_names.get(grupo, f"Grupo {grupo}")
        if grupo_nome not in agrupamento:
            agrupamento[grupo_nome] = []
        agrupamento[grupo_nome].append({
            "id": row[0],
            "titulo": row[1],
            "preco": row[2]
        })
    return agrupamento
