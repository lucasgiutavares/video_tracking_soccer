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
| `<nome>_detections.csv` | `frame, timestamp, class_id, class_name, confidence, x1, y1, x2, y2, width, height, center_x, center_y, area` (coordenadas no quadro original) |

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
