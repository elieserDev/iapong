import sqlite3
import pandas as pd


# time.sleep(2)

# Criar uma conexão com o banco de dados SQLite
conexao = sqlite3.connect('ia.db')

# Criando as colunas
column_names = ['acertou', 'errou', 'timeseries']

# Substitua 'seu_arquivo.txt' pelo caminho do seu arquivo
data = pd.read_csv('performance.txt', sep=' ', header=None, names=column_names)  # Separe por tabulação ('\t') ou outro separador, se necessário

# Inserir os dados do DataFrame na tabela 'minhatabela'
tabela_nome = 'performance'
data.to_sql(tabela_nome, conexao, if_exists='replace', index=False)

print('Dados inseridos no banco SQLite!')

# Fechar a conexão
conexao.close()
