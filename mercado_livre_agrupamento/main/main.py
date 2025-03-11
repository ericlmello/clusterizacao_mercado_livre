
# Tech chalenger - Eric lopes - FIAP
#Pesquisa de valores atuais de produtos Mercado Livre

#Salva dados em um BD SQLite
#Realiza o agrupamento com kmeans
#Cria uma nova coluna no BD com a grupo do produto (segmentação)
#Plota um gráfico com os clusters e a marcação dos outlaires 



######################################################################


from flask import Flask, jsonify, render_template
import requests
import sqlite3
import os
import logging
from datetime import datetime
from ml_utils import perform_kmeans_clustering, create_price_cluster_plot, organize_clusters

SEARCH_TERM = input("Digite o termo de pesquisa: ")

# Configurações

API_URL = f"https://api.mercadolibre.com/sites/MLB/search?q={SEARCH_TERM.replace(' ', '%20')}&limit=50"
DB_PATH = r"C:\Users\e178454_uss\Desktop\mercado_livre.db"

# Configuração de logs
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Criar a aplicação Flask
app = Flask(__name__, template_folder="templates")

# Criar banco de dados SQLite
def setup_database():
    logging.info("Criando/verificando banco de dados...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS produtos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT,
            titulo TEXT,
            preco REAL
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Banco de dados pronto.")

# Rota para coletar preços do Mercado Livre e salvar no banco
@app.route('/buscar', methods=['GET'])
def fetch_prices():
    logging.info("Buscando preços no Mercado Livre...")
    response = requests.get(API_URL)
    
    if response.status_code != 200:
        logging.error(f"Erro na API do Mercado Livre: {response.status_code}")
        return jsonify({"erro": "Erro ao buscar dados"}), 500

    data = response.json()
    
    if "results" in data:
        products = [
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), item["title"], item["price"])
            for item in data["results"]
        ]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO produtos (data, titulo, preco) VALUES (?, ?, ?)", products)
        conn.commit()
        conn.close()

        logging.info(f"{len(products)} preços coletados e salvos no banco.")
        return jsonify({"mensagem": f"{len(products)} preços coletados e salvos no banco."}), 200
    else:
        logging.error("Erro: resposta da API não contém 'results'.")
        return jsonify({"erro": "Erro ao buscar dados"}), 500

# Rota para consultar os dados armazenados
@app.route('/dados', methods=['GET'])
def consultar_dados():
    logging.info("Consultando dados do banco...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM produtos")
    rows = cursor.fetchall()
    conn.close()

    produtos = [{"id": row[0], "data": row[1], "titulo": row[2], "preco": row[3]} for row in rows]
    logging.info(f"{len(produtos)} registros encontrados.")
    
    return jsonify(produtos)

@app.route('/agrupamento', methods=['GET'])
def agrupar_precos():
    logging.info("Iniciando agrupamento de preços...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, titulo, preco FROM produtos")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logging.warning("Nenhum dado encontrado para agrupar.")
        return jsonify({"erro": "Nenhum dado encontrado para agrupar."}), 404

    precos = [row[2] for row in rows]
    precos_unicos = set(precos)

    if len(precos_unicos) < 3:
        logging.warning("Dados insuficientes para agrupar: menos de 3 preços distintos.")
        return jsonify({"erro": "Dados insuficientes para agrupamento. Pelo menos 3 preços distintos são necessários."}), 400

  
    clusters, kmeans = perform_kmeans_clustering(precos)
    if clusters is None:
        return jsonify({"erro": "Falha no agrupamento"}), 500

   
    agrupamento = organize_clusters(rows, clusters)

    
    graph_path = create_price_cluster_plot(rows, clusters)
    if graph_path is None:
        return jsonify({"erro": "Falha ao gerar gráfico"}), 500

    return render_template("agrupamento.html", imagem_url=graph_path, agrupamento=agrupamento)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    setup_database()
    logging.info("Iniciando servidor Flask...")
    app.run(host='0.0.0.0', port=5000, use_reloader=False)
