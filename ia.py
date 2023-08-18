import os
import time
import numpy as np
import pygame
import random
import sqlite3
import pandas as pd
from random import uniform

# Criando pesos aleatórios para começar o desenvolvimento dos neuronios
pesosPrimeiroNeuronio = np.array([uniform(-1,1) for i in range(4)])
pesosSegundoNeuronio = np.array([uniform(-1,1) for i in range(4)])
pesosTerceiroNeuronio = np.array([uniform(-1, 1) for i in range(2)])
pesosQuartoNeuronio = np.array([uniform(-1, 1) for i in range(2)])
pesosNeuronioSaida = np.array([uniform(-1,1) for i in range(2)])

def tangentHiperbolica(x):
    th = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return th

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Entradas da rede neural
def create_input_nn(raquete_x, bola_x, bola_y):
    bias = 1
    inputs = np.array([raquete_x, bola_x, bola_y, bias])
    return inputs

def feedforward(inputs, pesosPrimeiroNeuronio, pesosSegundoNeuronio, pesosTerceiroNeuronio, pesosQuartoNeuronio, pesosNeuronioSaida):
    global saidaPrimeiroNeuronio
    global saidaSegundoNeuronio
    global saidaTerceiroNeuronio
    global saidaQuartoNeuronio
    global resultado
    saidaPrimeiroNeuronio = round(tangentHiperbolica(np.sum(inputs * pesosPrimeiroNeuronio)), 6)
    saidaSegundoNeuronio = round(tangentHiperbolica(np.sum(inputs * pesosSegundoNeuronio)), 6)
    saidaTerceiroNeuronio = round(tangentHiperbolica(np.sum(np.array([saidaPrimeiroNeuronio, saidaSegundoNeuronio]) * pesosTerceiroNeuronio)), 6)
    saidaQuartoNeuronio = round(tangentHiperbolica(np.sum(np.array([saidaPrimeiroNeuronio, saidaSegundoNeuronio]) * pesosQuartoNeuronio)), 6)
    resultado = round(sigmoid(np.sum(np.array([saidaTerceiroNeuronio, saidaQuartoNeuronio]) * pesosNeuronioSaida)), 6)

    return resultado

def backpropagation(inputs, erro, pesosPrimeiroNeuronio, pesosSegundoNeuronio, pesosTerceiroNeuronio, pesosQuartoNeuronio, pesosNeuronioSaida):
    alpha = 0.1

    for i in range(len(pesosNeuronioSaida)):
        if i == 0:
            entrada = saidaTerceiroNeuronio
        elif i == 1:
            entrada = saidaQuartoNeuronio
        
        pesosNeuronioSaida[i] = pesosNeuronioSaida[i] + (alpha * entrada * erro)

    for i in range(len(pesosQuartoNeuronio)):
        if i == 0:
            entrada1 = saidaPrimeiroNeuronio
        elif i == 0:
            entrada1 = saidaSegundoNeuronio

        pesosQuartoNeuronio[i] = pesosQuartoNeuronio[i] + (alpha * entrada1 * erro)

    for i in range(len(pesosTerceiroNeuronio)):
        if i == 0:
            entrada2 = saidaPrimeiroNeuronio
        elif i == 0:
            entrada2 = saidaSegundoNeuronio

        pesosTerceiroNeuronio[i] = pesosTerceiroNeuronio[i] + (alpha * entrada2 * erro)

    for i in range(len(pesosSegundoNeuronio)):
        pesosSegundoNeuronio[i] = pesosSegundoNeuronio[i] + (alpha * inputs[i] * erro)
    
    for i in range(len(pesosPrimeiroNeuronio)):
        pesosPrimeiroNeuronio[i] = pesosPrimeiroNeuronio[i] + (alpha * inputs[i] * erro)

    print(resultado)

def save_db():
    # Criar uma conexão com o banco de dados SQLite
    conexao = sqlite3.connect('ia.db')

    # Criando as colunas
    column_names = ['acertou', 'errou', 'timeseries', 'key']

    # Substitua 'seu_arquivo.txt' pelo caminho do seu arquivo
    data = pd.read_csv('performance.txt', sep=' ', header=None, names=column_names)  # Separe por tabulação ('\t') ou outro separador, se necessário

    # Inserir os dados do DataFrame na tabela 'minhatabela'
    tabela_nome = 'performance'
    data.to_sql(tabela_nome, conexao, if_exists='replace', index=False)

    print('Dados inseridos no banco SQLite!')

    # Fechar a conexão
    conexao.close()


# Inicialização do Pygame
pygame.init()

# Configurações da tela
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('IA Pong')
sound_error = pygame.mixer.Sound('error.mp3')
sound_yes = pygame.mixer.Sound('yes.mp3')
sound_wall = pygame.mixer.Sound('wallsong.mp3')

# Cores
ui = (0, 255, 0)
player = (55, 255, 55)
ball = (255, 55, 55)
black = (15, 35, 23)

# Configurações da raquete
raquete_width = 80
raquete_height = 10
raquete_x = (screen_width - raquete_width) // 2
raquete_y = screen_height - raquete_height - 5
raquete_speed = 20

# Configurações da bola
bola_width = 15
bola_height = 15
bola_x = random.randint(0, screen_width - bola_width)
bola_y = random.randint(bola_height, screen_height // 2)
bola_speed_x = 9
bola_speed_y = 9

# Placar
defendeu = 0
errou = 0
start_time = time.time()
font = pygame.font.Font(None, 24)

# Loop principal do jogo
running = True
clock = pygame.time.Clock()

# Instanciando nome do arquivo de dados
trainingDataFile = 'trainingData.txt'
performanceFile = 'performance.txt'

while running:
    inputs_nn = create_input_nn(raquete_x/800, bola_x/800, bola_y/600)
    key = feedforward(inputs_nn, pesosPrimeiroNeuronio, pesosSegundoNeuronio, pesosTerceiroNeuronio, pesosQuartoNeuronio, pesosNeuronioSaida)

    with open(trainingDataFile, 'a') as arquivo:
        arquivo.write(str(raquete_x) + " " + str(bola_x) + " " + str(bola_y) + " " + str(key) + "\n")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if key > 0.5 and raquete_x > 0:
        raquete_x -= raquete_speed
    if key < 0.5 and raquete_x < screen_width - raquete_width:
        raquete_x += raquete_speed

    # Atualização da posição da bola
    bola_x += bola_speed_x
    bola_y += bola_speed_y

    # Verificação de colisões com as paredes
    if bola_x <= 0.1 or bola_x >= screen_width - bola_width:
        sound_wall.play()
        bola_speed_x *= -1
    if bola_y <= 0.1:
        sound_wall.play()
        bola_speed_y *= -1
    if bola_y >= (screen_height - bola_height):
        bola_x = screen_width // 2
        bola_y = screen_height // 6
        bola_speed_x *= random.choice([1, -1])
        bola_speed_y *= random.choice([1, -1])
        sound_error.play()
        errou += 1

    # Verificação de colisão com a raquete
    if raquete_x < bola_x + bola_width and raquete_x + raquete_width > bola_x and raquete_y-2 < bola_y + bola_height and raquete_y-2 + raquete_height > bola_y:
        bola_speed_y *= -1
        sound_yes.play()
        defendeu += 1

    # Desenho na tela
    screen.fill(black)
    pygame.draw.rect(screen, player, (raquete_x, raquete_y, raquete_width, raquete_height))
    pygame.draw.ellipse(screen, ball, (bola_x, bola_y, bola_width, bola_height))

    #Cronometro
    elapsed_time = time.time() - start_time
    text_time = "Tempo decorrido: {:.2f} segundos".format(elapsed_time)
    time_value = float("{:.2f}".format(elapsed_time))

    # Renderização do placar
    text = font.render(f'Defendeu: {defendeu}', True, ui)
    screen.blit(text, (10, 10))
    text2 = font.render(f'Errou: {errou}', True, ui)
    screen.blit(text2, (10, 30))
    text3 = font.render(text_time, True, ui)
    screen.blit(text3, (10, 60))

    # Criando registros de performance
    with open(performanceFile, 'a') as arquivo:
        arquivo.write(str(defendeu) + " " + str(errou) + " " + str(time_value) + " " + str(key) + "\n")

    #Calculando o erro e chamando o backpropagation
    erro = (raquete_x - bola_x) / 1000
    backpropagation(inputs_nn, 
                    erro, 
                    pesosPrimeiroNeuronio, 
                    pesosSegundoNeuronio, 
                    pesosTerceiroNeuronio, 
                    pesosQuartoNeuronio, 
                    pesosNeuronioSaida)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
save_db()

# Apaga os registros gerados
if os.path.exists(trainingDataFile):
    os.remove(trainingDataFile)
if os.path.exists(performanceFile):
    os.remove(performanceFile)