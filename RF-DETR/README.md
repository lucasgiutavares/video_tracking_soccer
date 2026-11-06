# Detecção de objetos em futebol — RF-DETR SoccerNet

Detecta **bola, jogadores, árbitros e goleiros** em vídeos usando o modelo
[julianzu9612/RFDETR-Soccernet](https://huggingface.co/julianzu9612/RFDETR-Soccernet)
(RF-DETR-Large, licença Apache-2.0).

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Os pesos (~1,4 GB) são baixados do Hugging Face na primeira execução e ficam em cache.
Se preferir, baixe o `.pth` manualmente e passe `--weights caminho/do/arquivo.pth`.
Também é necessário ter o `ffmpeg` instalado para gerar o vídeo em H.264; sem ele, o script grava com o OpenCV (mp4v).

## Uso

```bash
# vídeo inteiro, todos os quadros
python detect_soccer.py teste_corte23.mp4

# teste rápido: 1 a cada 5 quadros, primeiros 10 s
python detect_soccer.py teste_corte23.mp4 --frame-skip 5 --end 10

# 4K com câmera aberta: blocos 2x2 para achar bola/jogadores pequenos
python detect_soccer.py teste_corte23.mp4 --tiles 2 --single-ball \
    --class-threshold ball=0.3 --class-threshold player=0.6
```

Saídas em `./saida/` (mude com `-o`):

| arquivo | conteúdo |
|---|---|
| `<nome>_annotated.mp4` | vídeo com caixas (bola marcada com triângulo), 1920 px de largura por padrão |
| `<nome>_detections.csv` | `frame, timestamp, class_id, class_name, confidence, x1, y1, x2, y2, width, height, center_x, center_y, area, team` (coordenadas no quadro original; `team` = A/B para jogadores e goleiros) |
| `<nome>_possession.csv` | por quadro analisado: `frame, timestamp, ball_detected, team_near_ball, possession_team, seconds_A, seconds_B, pct_A, pct_B` |

## Posse de bola

Segue a ideia do `analyze_ball_possession(df, distance_threshold=100)` do model card:
o jogador mais próximo da bola, até `--possession-distance` px, tem a posse.

1. **Equipes:** o modelo não distingue equipes. Antes do processamento, o script amostra
   `--team-calib-frames` quadros, extrai a cor do tronco de cada jogador (descartando grama) e
   separa as cores em 2 grupos (k-means). Equipe A = camisa mais clara. Goleiros vão para a equipe
   cujos jogadores estão mais perto deles no eixo horizontal. Árbitros ficam de fora.
2. **Posse:** a posse fica com a última equipe que teve a bola e só troca quando o modelo vê a
   outra equipe com ela por pelo menos `--possession-min-seconds` segundos seguidos (padrão 3).
   Enquanto a bola não é identificada (ou está longe de todos), a posse continua com a última
   equipe e a contagem da troca fica pausada. A contagem só zera quando a equipe com a posse
   volta a ser vista com a bola. O tempo antes da primeira posse não é contado.
3. **Vídeo:** caixas na cor de cada equipe, elipse vermelha no jogador com a bola e um painel no
   canto inferior esquerdo com o tempo (mm:ss), a % de cada equipe e uma barra proporcional.

| opção | efeito |
|---|---|
| `--possession-distance 100` | distância máx. bola–caixa do jogador, em px de um quadro 1080p (escala com a resolução) |
| `--possession-min-seconds 3` | segundos seguidos com a outra equipe para a posse trocar |
| `--team-calib-frames 10` | quadros extras analisados para aprender as cores |
| `--no-possession` | desativa equipes e posse |

Dica: use `--single-ball` e, em 4K, `--tiles 2` com `--class-threshold ball=0.3`, porque a posse depende de achar a bola.

## Campo e homografia (`--pitch`)

Usa o modelo de pontos-chave do campo [Soccana](https://huggingface.co/Adit-jain/Soccana_Keypoint)
(YOLO11n-pose treinado no SoccerNet-Calibration). Ele encontra até 29 pontos de referência:
cantos, áreas grande e pequena, vértice da meia-lua, linha do meio e círculo central. Com pelo
menos 4 pontos confiáveis e não alinhados, o script calcula a homografia que leva cada pixel para
metros num campo FIFA padrão de 105 × 68 m. A homografia é recalculada a cada quadro analisado,
porque a câmera se move. Os pesos são baixados do Hugging Face na primeira execução.

```bash
pip install ultralytics
python detect_soccer.py teste_corte23.mp4 --pitch --pitch-show-keypoints

# pesos já baixados
python detect_soccer.py teste_corte23.mp4 --pitch --pitch-weights best.pt
```

Saídas com `--pitch`:

- `pitch_x, pitch_y` no `<nome>_detections.csv`: posição em metros no campo. Para pessoas é o ponto
  dos pés; para a bola é o centro da caixa, então uma bola no alto cai no gramado atrás dela.
- Filtro de velocidade (jogadores, goleiros e árbitros): cada atleta é seguido de um quadro ao
  outro pela caixa na imagem (`track_id`). Se a posição pular mais rápido que `--max-speed`
  (11 m/s) em relação à última posição válida, ela é removida e substituída pela interpolação
  linear até a próxima posição válida (`pitch_flag = interpolado`). Se nenhuma posição válida
  chegar em `--max-gap` segundos, fica vazia (`pitch_flag = removido`). `pitch_x_raw, pitch_y_raw`
  guardam o valor original da homografia. O minimapa usa as posições já filtradas, por isso o
  vídeo e o CSV são gravados com um atraso de até `--max-gap` segundos.
- `<nome>_pitch.csv`: por quadro analisado, `frame, timestamp, keypoints, homography, fresh, h11..h33`.
  `homography = 1` quando há homografia válida e `fresh = 0` quando ela foi herdada do quadro anterior.
- Painel **POSICIONAMENTO** no canto inferior direito do vídeo: vista de cima do campo com a
  posição dos atletas. Jogadores são círculos na cor da equipe, goleiros são quadrados, árbitros
  são círculos pretos e a bola é amarela. O jogador com a bola tem contorno vermelho. Uma legenda
  fica abaixo do campo. Quando o campo não é identificado no quadro, o painel avisa.

| opção | efeito |
|---|---|
| `--pitch` | ativa a detecção do campo e a homografia |
| `--pitch-weights arq.pt` | pesos locais do modelo de campo (padrão: baixa do Hugging Face) |
| `--pitch-imgsz 640` | resolução de entrada do modelo de campo (a do treino) |
| `--pitch-kp-conf 0.5` | confiança mínima de cada ponto do campo |
| `--pitch-max-age 0.5` | segundos que a última homografia vale quando faltam pontos |
| `--pitch-show-keypoints` | desenha os pontos detectados no vídeo, para conferir o modelo |
| `--no-radar` | não desenha o minimapa |
| `--max-speed 11` | velocidade máx. de um atleta (m/s); acima disso a posição é interpolada (0 desativa) |
| `--max-gap 1.0` | segundos máx. esperando a próxima posição válida para interpolar |
| `--track-match-dist 1.0` | distância máx. dos pés entre quadros analisados, em alturas de caixa, para ser o mesmo atleta |

Limitações: o campo de referência tem tamanho fixo, então as distâncias são aproximadas em campos
de outro tamanho. Em planos fechados ou replays, poucos pontos aparecem e a homografia fica indisponível.

## Opções principais

| opção | efeito |
|---|---|
| `--threshold 0.5` | confiança mínima geral (model card: 0.5 geral, ≥0.7 alta precisão) |
| `--class-threshold CLASSE=V` | limiar por classe (`ball`, `player`, `referee`, `goalkeeper`) |
| `--frame-skip N` | analisa 1 a cada N quadros; com `--reuse-detections` o vídeo mantém todos os quadros |
| `--tiles N` | divide o quadro em N×N blocos + quadro inteiro e une com NMS; custa N²+1 inferências |
| `--single-ball` | mantém só a bola de maior confiança por quadro |
| `--device cuda` / `--optimize` | GPU e otimização de inferência |
| `--resolution 1280` | força a resolução de entrada (o model card usa 1280) |
| `--class-names a,b,c,d` | redefine os nomes se os rótulos saírem trocados |
| `--no-video` | só o CSV |

## Observações

- **Desempenho:** é um modelo de 128 M de parâmetros. Em GPU, roda em tempo próximo do real; em CPU, leva vários segundos por quadro. Para testar em CPU, use `--frame-skip` e `--end`.
- **Nomes das classes:** o script usa os nomes gravados no checkpoint. Se o checkpoint não os tiver, assume `ball, player, referee, goalkeeper` (IDs 0–3), porque o model card não mapeia cada ID. Na primeira execução, confira no vídeo anotado se os rótulos batem e, se não baterem, corrija com `--class-names`.
- **Carregamento:** o checkpoint é carregado com `trust_checkpoint=True`, que é necessário para arquivos salvos por versões antigas do `rfdetr`. Use somente com pesos de origem confiável.
