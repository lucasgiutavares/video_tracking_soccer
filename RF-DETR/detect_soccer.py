#!/usr/bin/env python3
"""
Detecção de objetos em vídeos de futebol com RF-DETR (SoccerNet).

Modelo: https://huggingface.co/julianzu9612/RFDETR-Soccernet
        RF-DETR-Large fine-tuned no SoccerNet-Tracking — classes: bola, jogador,
        árbitro, goleiro.

Saídas:
  * vídeo anotado (caixas + rótulos; a bola é marcada com um triângulo)
  * CSV com uma linha por detecção (frame, tempo, classe, caixa, confiança...)
  * resumo por classe no terminal

Exemplos:
  python detect_soccer.py teste_corte23.mp4
  python detect_soccer.py teste_corte23.mp4 --frame-skip 5 --max-frames 300
  python detect_soccer.py jogo.mp4 --weights ./checkpoint.pth --device cuda \
      --class-threshold player=0.7 --class-threshold ball=0.4
  python detect_soccer.py teste_corte23.mp4 --pitch   # homografia do campo (modelo Soccana)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HF_REPO = "julianzu9612/RFDETR-Soccernet"

# Ordem de classes usada quando o checkpoint não traz os nomes embutidos.
# O model card informa IDs 0-3 sem mapear cada um; confira com --class-names
# se os rótulos saírem trocados.
DEFAULT_CLASS_NAMES = ["ball", "player", "referee", "goalkeeper"]

# Cores BGR por classe
CLASS_COLORS = {
    "ball": (0, 255, 255),        # amarelo
    "player": (255, 140, 0),      # azul
    "referee": (40, 40, 40),      # preto
    "goalkeeper": (60, 200, 60),  # verde
}
FALLBACK_COLOR = (200, 0, 200)


# --------------------------------------------------------------------------- #
# Detecções
# --------------------------------------------------------------------------- #
@dataclass
class Detections:
    xyxy: np.ndarray        # (N, 4) float, coordenadas no quadro original
    confidence: np.ndarray  # (N,)
    class_id: np.ndarray    # (N,) int

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))

    def __len__(self) -> int:
        return len(self.class_id)

    def filter(self, mask: np.ndarray) -> "Detections":
        return Detections(self.xyxy[mask], self.confidence[mask], self.class_id[mask])


class SoccerDetector:
    """Envolve o RF-DETR; recebe quadros BGR (OpenCV) e devolve Detections."""

    def __init__(self, weights: str | None, device: str | None, resolution: int | None,
                 class_names: list[str] | None, optimize: bool):
        try:
            import warnings
            # O rfdetr compila funções de perda (só usadas no treino) com torch.jit.script
            # ao ser importado; versões recentes do PyTorch emitem um FutureWarning inofensivo.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated",
                                        category=FutureWarning)
                from rfdetr.detr import RFDETR
        except ImportError:
            sys.exit("Pacote 'rfdetr' não encontrado. Instale com: pip install -r requirements.txt")

        weights_path = weights or download_weights()
        kwargs = {}
        if device:
            kwargs["device"] = device
        if resolution:
            kwargs["resolution"] = resolution

        print(f"[modelo] carregando {weights_path}")
        # trust_checkpoint=True é necessário para checkpoints salvos por versões
        # antigas do rfdetr (contêm objetos Python além dos tensores) — só use com
        # arquivos confiáveis.
        try:
            self.model = RFDETR.from_checkpoint(weights_path, trust_checkpoint=True, **kwargs)
        except ValueError as exc:
            if "patch_size" not in str(exc):
                raise
            # O checkpoint do SoccerNet foi treinado com o RF-DETR Large antigo
            # (DINOv2-base, patch 14). Nas versões novas do rfdetr, "RFDETRLarge" é
            # outra arquitetura (patch 16) e o from_checkpoint escolhe a errada.
            self.model = load_legacy_large(weights_path, device, resolution)

        if optimize:
            try:
                self.model.optimize_for_inference()
                print("[modelo] otimizado para inferência")
            except Exception as exc:  # noqa: BLE001 - otimização é opcional
                print(f"[modelo] optimize_for_inference indisponível ({exc}); seguindo sem")

        self.class_names = class_names or self._names_from_checkpoint()
        print(f"[modelo] classes: {self.class_names}")

    def _names_from_checkpoint(self) -> list[str]:
        try:
            names = list(self.model.class_names)
        except Exception:  # noqa: BLE001
            names = []
        # Sem nomes embutidos o rfdetr devolve as 80 classes do COCO.
        if not names or len(names) >= 80:
            return DEFAULT_CLASS_NAMES
        # Datasets exportados pelo Roboflow às vezes trazem uma superclasse na
        # posição 0 (ex.: "players-ball-referee"); ela nunca é prevista.
        return [n.lower() for n in names]

    def __call__(self, frame_bgr: np.ndarray, threshold: float) -> Detections:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=threshold, include_source_image=False)
        if det is None or len(det) == 0:
            return Detections.empty()
        return Detections(
            np.asarray(det.xyxy, np.float32),
            np.asarray(det.confidence, np.float32),
            np.asarray(det.class_id, int),
        )


def checkpoint_args(path: str) -> dict:
    """Lê os hiperparâmetros gravados no checkpoint (sem carregar tudo na RAM)."""
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 - formato antigo sem suporte a mmap
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    out = dict(args) if isinstance(args, dict) else dict(vars(args)) if args is not None else {}
    del ckpt
    return out


def load_legacy_large(path: str, device: str | None, resolution: int | None):
    """Carrega o checkpoint com a arquitetura RF-DETR Large legada (DINOv2-base, patch 14)."""
    import warnings
    from rfdetr import RFDETRLargeDeprecated

    args = checkpoint_args(path)
    patch, windows = int(args.get("patch_size", 14)), int(args.get("num_windows", 4))
    step = patch * windows
    res = resolution or args.get("resolution") or 1288
    if res % step:
        fixed = max(step, round(res / step) * step)
        print(f"[modelo] resolução {res} não é múltipla de {step}; usando {fixed}")
        res = fixed
    kwargs = {"pretrain_weights": path, "resolution": int(res)}
    if device:
        kwargs["device"] = device
    print(f"[modelo] arquitetura RF-DETR Large legada (patch {patch}), resolução {res}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # num_classes não é informado: o rfdetr usa o do checkpoint (4 classes).
        return RFDETRLargeDeprecated(trust_checkpoint=True, **kwargs)


def nms_per_class(det: Detections, iou_thr: float = 0.5) -> Detections:
    """Supressão de não-máximos por classe (remove duplicatas na sobreposição dos blocos)."""
    if len(det) == 0:
        return det
    keep_all = []
    for cid in np.unique(det.class_id):
        idx = np.flatnonzero(det.class_id == cid)
        idx = idx[np.argsort(-det.confidence[idx])]
        boxes = det.xyxy
        while idx.size:
            i, rest = idx[0], idx[1:]
            keep_all.append(i)
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
            # IoU, mas também descarta caixa quase toda contida em outra (objeto cortado na borda do bloco)
            iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-6)
            ios = inter / (np.minimum(area(boxes[i]), area(boxes[rest])) + 1e-6)
            idx = rest[(iou < iou_thr) & (ios < 0.8)]
    keep = np.zeros(len(det), bool)
    keep[keep_all] = True
    return det.filter(keep)


def detect_tiled(detector, frame: np.ndarray, threshold: float, grid: int, overlap: float) -> Detections:
    """Roda o detector em uma grade grid x grid de blocos sobrepostos + no quadro inteiro.

    Útil em vídeos 4K com câmera aberta: o modelo reduz a imagem para ~1280 px,
    e a bola/jogadores distantes ficam pequenos demais. Custo: grid² + 1 inferências.
    """
    H, W = frame.shape[:2]
    parts = [detector(frame, threshold)]
    tw, th = int(np.ceil(W / (grid - (grid - 1) * overlap))), int(np.ceil(H / (grid - (grid - 1) * overlap)))
    xs = np.linspace(0, W - tw, grid).astype(int)
    ys = np.linspace(0, H - th, grid).astype(int)
    for y0 in ys:
        for x0 in xs:
            d = detector(frame[y0:y0 + th, x0:x0 + tw], threshold)
            if len(d):
                d.xyxy = d.xyxy + np.array([x0, y0, x0, y0], np.float32)
                parts.append(d)
    merged = Detections(np.concatenate([p.xyxy for p in parts]),
                        np.concatenate([p.confidence for p in parts]),
                        np.concatenate([p.class_id for p in parts]))
    return nms_per_class(merged)


def download_weights() -> str:
    """Baixa o checkpoint .pth do Hugging Face (fica em cache após a 1ª vez)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        sys.exit("Instale huggingface_hub ou informe --weights com o caminho do .pth")
    files = [f for f in list_repo_files(HF_REPO) if f.endswith((".pth", ".pt"))]
    if not files:
        sys.exit(f"Nenhum arquivo .pth encontrado em {HF_REPO}; use --weights")
    # Prioriza o melhor checkpoint (EMA / best) quando houver vários.
    files.sort(key=lambda f: (("best" not in f.lower()), ("ema" not in f.lower()), len(f)))
    print(f"[modelo] baixando {files[0]} de {HF_REPO} (~1,4 GB na primeira vez)")
    return hf_hub_download(HF_REPO, files[0])


# --------------------------------------------------------------------------- #
# Desenho
# --------------------------------------------------------------------------- #
LABELS_PT = {"ball": "bola", "player": "jogador", "referee": "arbitro", "goalkeeper": "goleiro"}


def class_name(names: list[str], cid: int) -> str:
    return names[cid] if 0 <= cid < len(names) else f"class_{cid}"


def text_color_for(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Preto ou branco, o que tiver mais contraste com a cor de fundo."""
    b, g, r = bgr
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255, 255, 255)


def annotate(frame: np.ndarray, det: Detections, names: list[str], show_conf: bool = True,
             teams: np.ndarray | None = None, team_colors: list | None = None,
             holder: int | None = None) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    thick = max(2, h // 540)
    font_scale = max(0.5, h / 1600)
    for i, ((x1, y1, x2, y2), conf, cid) in enumerate(zip(det.xyxy, det.confidence, det.class_id)):
        name = class_name(names, int(cid))
        color = CLASS_COLORS.get(name, FALLBACK_COLOR)
        team = int(teams[i]) if teams is not None and i < len(teams) else -1
        if team >= 0 and team_colors:
            color = team_colors[team]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        if i == holder:
            # jogador com a bola: elipse vermelha nos pés
            cv2.ellipse(out, ((x1 + x2) // 2, y2), (max(12, (x2 - x1) // 2 + 6), max(5, (x2 - x1) // 5 + 3)),
                        0, 0, 360, (0, 0, 255), thick + 1, cv2.LINE_AA)
        if name == "ball":
            # triângulo acima da bola — mais visível que uma caixa minúscula
            cx, s = (x1 + x2) // 2, max(10, h // 90)
            pts = np.array([[cx, y1 - 4], [cx - s, y1 - 4 - 2 * s], [cx + s, y1 - 4 - 2 * s]])
            cv2.fillPoly(out, [pts], color)
            cv2.polylines(out, [pts], True, (0, 0, 0), max(1, thick // 2))
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = LABELS_PT.get(name, name) + (f" {TEAM_SHORT[team]}" if team >= 0 else "") \
            + (f" {conf:.2f}" if show_conf else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        if team >= 0:
            txt_color = text_color_for(color)
        else:
            txt_color = (255, 255, 255) if name in ("referee", "player") else (0, 0, 0)
        cv2.putText(out, label, (x1 + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    txt_color, thick, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Equipes e posse de bola
# --------------------------------------------------------------------------- #
# O modelo não distingue equipes: elas são separadas pela cor da camisa
# (k-means com 2 grupos sobre a cor do tronco de cada jogador).
TEAM_NAMES = ["Equipe A", "Equipe B"]
TEAM_SHORT = ["A", "B"]
# Peso dos canais L, a, b no agrupamento — o brilho varia com sombra/iluminação,
# então pesa menos que a tonalidade.
LAB_WEIGHTS = np.array([0.5, 1.0, 1.0], np.float32)


def jersey_feature(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Cor mediana (Lab) do tronco do jogador, ignorando pixels de grama."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return None
    H, W = frame.shape[:2]
    cx1, cx2 = int(max(0, x1 + 0.2 * w)), int(min(W, x2 - 0.2 * w))
    cy1, cy2 = int(max(0, y1 + 0.15 * h)), int(min(H, y1 + 0.5 * h))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 50) & (hsv[:, 2] > 40)
    px = lab[~grass]
    if len(px) < max(4, 0.15 * len(lab)):
        px = lab  # camisa verde (ou recorte só de grama): usa tudo
    return np.median(px, axis=0).astype(np.float32)


class TeamClassifier:
    """Agrupa jogadores em 2 equipes pela cor da camisa."""

    def __init__(self):
        self.centers: np.ndarray | None = None
        self.colors: list[tuple[int, int, int]] = [(255, 255, 255), (0, 0, 255)]

    @property
    def fitted(self) -> bool:
        return self.centers is not None

    def fit(self, feats: list[np.ndarray]) -> None:
        data = np.stack(feats).astype(np.float32) * LAB_WEIGHTS
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centers = cv2.kmeans(data, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        # ordem estável entre execuções: equipe A = camisa mais clara
        self.centers = centers[np.argsort(-centers[:, 0])]
        lab = (self.centers / LAB_WEIGHTS).clip(0, 255).astype(np.uint8).reshape(1, 2, 3)
        self.colors = [tuple(int(c) for c in v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0]]

    def predict(self, frame: np.ndarray, det: Detections, names: list[str]) -> np.ndarray:
        """Equipe (0/1) de cada detecção; -1 para bola, árbitro ou indefinido."""
        teams = np.full(len(det), -1, int)
        if not self.fitted or len(det) == 0:
            return teams
        cls = [class_name(names, int(c)) for c in det.class_id]
        for i, name in enumerate(cls):
            if name == "player":
                f = jersey_feature(frame, det.xyxy[i])
                if f is not None:
                    teams[i] = int(np.argmin(np.linalg.norm(self.centers - f * LAB_WEIGHTS, axis=1)))
        # Goleiro usa uniforme diferente: vai para a equipe cujos jogadores estão,
        # em média, mais perto dele no eixo horizontal (ele fica no lado da sua equipe).
        cx = (det.xyxy[:, 0] + det.xyxy[:, 2]) / 2
        means = [cx[teams == t].mean() if (teams == t).any() else None for t in (0, 1)]
        for i, name in enumerate(cls):
            if name == "goalkeeper":
                opts = [(abs(cx[i] - m), t) for t, m in enumerate(means) if m is not None]
                if opts:
                    teams[i] = min(opts)[1]
        return teams


class PossessionTracker:
    """Posse de bola: o jogador mais próximo da bola (dentro de `max_dist` px) define a equipe.

    A posse fica com a última equipe que teve a bola. Ela só troca quando o modelo
    vê a outra equipe com a bola por pelo menos `min_seconds` segundos seguidos
    (evita trocas por ruído). Enquanto a bola não é identificada (ou está longe de
    todos), a posse continua com a última equipe e a contagem da troca fica pausada.
    A contagem só zera quando a equipe com a posse volta a ser vista com a bola.
    Antes da primeira posse, o tempo não é contado para ninguém.
    """

    def __init__(self, max_dist: float, min_seconds: float):
        self.max_dist = max_dist
        self.min_seconds = max(0.0, min_seconds)
        self.team = -1
        self.candidate, self.streak = -1, 0.0  # equipe desafiante e segundos seguidos com a bola
        self.seconds = [0.0, 0.0]
        self.holder: int | None = None

    def update(self, det: Detections, teams: np.ndarray, names: list[str], dt: float) -> int:
        """Atualiza com um quadro analisado; devolve a equipe junto à bola (-1 se ninguém)."""
        self.holder = None
        near = -1
        is_ball = np.array([class_name(names, int(c)) == "ball" for c in det.class_id], bool)
        has_team = teams >= 0
        if is_ball.any() and has_team.any():
            b = np.flatnonzero(is_ball)[np.argmax(det.confidence[is_ball])]
            bx, by = (det.xyxy[b, 0] + det.xyxy[b, 2]) / 2, (det.xyxy[b, 1] + det.xyxy[b, 3]) / 2
            idx = np.flatnonzero(has_team)
            boxes = det.xyxy[idx]
            # distância do centro da bola até a caixa do jogador (0 se estiver dentro)
            dx = np.maximum.reduce([boxes[:, 0] - bx, np.zeros(len(idx)), bx - boxes[:, 2]])
            dy = np.maximum.reduce([boxes[:, 1] - by, np.zeros(len(idx)), by - boxes[:, 3]])
            d = np.hypot(dx, dy)
            j = int(np.argmin(d))
            if d[j] <= self.max_dist:
                self.holder = int(idx[j])
                near = int(teams[self.holder])
        if near >= 0:
            if self.team < 0:
                self.team = near  # primeira posse: assume na hora
            elif near == self.team:
                self.candidate, self.streak = -1, 0.0
            else:
                if near == self.candidate:
                    self.streak += dt
                else:
                    self.candidate, self.streak = near, dt
                if self.streak >= self.min_seconds - 1e-9:
                    self.team = near
                    self.candidate, self.streak = -1, 0.0
        if self.team >= 0:
            self.seconds[self.team] += dt
        return near

    def percentages(self) -> list[float]:
        total = sum(self.seconds)
        return [100 * s / total for s in self.seconds] if total > 0 else [0.0, 0.0]


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def draw_possession_panel(img: np.ndarray, tracker: PossessionTracker, colors: list) -> None:
    """Painel no canto inferior esquerdo: tempo e % de posse de cada equipe."""
    H = img.shape[0]
    s = H / 1080
    font = cv2.FONT_HERSHEY_SIMPLEX
    m, pw, ph = int(20 * s), int(430 * s), int(165 * s)
    x0, y0 = m, H - m - ph
    roi = img[y0:y0 + ph, x0:x0 + pw]
    roi[:] = cv2.addWeighted(roi, 0.35, np.full_like(roi, 20), 0.65, 0)
    t1, t2 = max(1, int(2 * s)), max(1, int(1.5 * s))
    cv2.putText(img, "POSSE DE BOLA", (x0 + int(14 * s), y0 + int(32 * s)), font, 0.75 * s,
                (255, 255, 255), t1, cv2.LINE_AA)
    pct = tracker.percentages()
    has_data = sum(tracker.seconds) > 0
    for t in (0, 1):
        ry = y0 + int((68 + 38 * t) * s)
        sw = int(22 * s)
        cv2.rectangle(img, (x0 + int(14 * s), ry - sw), (x0 + int(14 * s) + sw, ry), colors[t], -1)
        cv2.rectangle(img, (x0 + int(14 * s), ry - sw), (x0 + int(14 * s) + sw, ry), (255, 255, 255), 1)
        label = TEAM_NAMES[t] + ("  <" if tracker.team == t else "")
        cv2.putText(img, label, (x0 + int(46 * s), ry - int(3 * s)), font, 0.7 * s,
                    (255, 255, 255), t2, cv2.LINE_AA)
        cv2.putText(img, fmt_time(tracker.seconds[t]), (x0 + int(225 * s), ry - int(3 * s)), font,
                    0.7 * s, (255, 255, 255), t2, cv2.LINE_AA)
        p = f"{pct[t]:.0f}%" if has_data else "--"
        (tw, _), _ = cv2.getTextSize(p, font, 0.8 * s, t1)
        cv2.putText(img, p, (x0 + pw - int(14 * s) - tw, ry - int(2 * s)), font, 0.8 * s,
                    (255, 255, 255), t1, cv2.LINE_AA)
    # barra dividida pela porcentagem
    bx1, bx2 = x0 + int(14 * s), x0 + pw - int(14 * s)
    by1, by2 = y0 + ph - int(36 * s), y0 + ph - int(18 * s)
    if has_data:
        split = bx1 + int((bx2 - bx1) * pct[0] / 100)
        cv2.rectangle(img, (bx1, by1), (split, by2), colors[0], -1)
        cv2.rectangle(img, (split, by1), (bx2, by2), colors[1], -1)
    else:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (90, 90, 90), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)


def calibrate_teams(cap, detect, start: int, end: int, n_frames: int, names: list[str]) -> TeamClassifier:
    """Amostra quadros espalhados no trecho, coleta cores de camisa e ajusta o k-means."""
    feats = []
    for i in np.unique(np.linspace(start, end - 1, max(1, n_frames)).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect(frame)
        for box, cid in zip(det.xyxy, det.class_id):
            if class_name(names, int(cid)) == "player":
                f = jersey_feature(frame, box)
                if f is not None:
                    feats.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    clf = TeamClassifier()
    if len(feats) >= 6:
        clf.fit(feats)
    return clf


# --------------------------------------------------------------------------- #
# Campo e homografia
# --------------------------------------------------------------------------- #
# Modelo de pontos-chave do campo Soccana (YOLO11n-pose, 29 pontos, treinado no
# SoccerNet-Calibration): https://huggingface.co/Adit-jain/Soccana_Keypoint
PITCH_HF_REPO = "Adit-jain/Soccana_Keypoint"
PITCH_HF_FILE = "Model/weights/best.pt"

# Campo FIFA padrão em metros. x = comprimento (0 = linha de fundo da esquerda da
# imagem), y = largura (0 = linha lateral do fundo da imagem, "top" no modelo).
# Campos reais variam de tamanho; as coordenadas saem nesse campo de referência.
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PENALTY_BOX_W, PENALTY_BOX_L = 40.32, 16.5
GOAL_BOX_W, GOAL_BOX_L = 18.32, 5.5
CENTRE_CIRCLE_R, PENALTY_SPOT = 9.15, 11.0


def pitch_vertices() -> np.ndarray:
    """Posição no campo (m) de cada um dos 29 pontos-chave, na ordem de saída do modelo Soccana.

    "pt1" é o ponto na linha de fundo e "pt2" o ponto na linha da área. Os pontos
    10 e 26 são o vértice da meia-lua (marca do pênalti + raio de 9,15 m).
    """
    L, W = PITCH_LENGTH, PITCH_WIDTH
    pb1, pb2 = (W - PENALTY_BOX_W) / 2, (W + PENALTY_BOX_W) / 2
    gb1, gb2 = (W - GOAL_BOX_W) / 2, (W + GOAL_BOX_W) / 2
    arc = PENALTY_SPOT + CENTRE_CIRCLE_R
    r = CENTRE_CIRCLE_R
    return np.array([
        (0, 0),                                   # 0  sideline_top_left
        (0, pb1), (PENALTY_BOX_L, pb1),           # 1  big_rect_left_top_pt1, 2 pt2
        (0, pb2), (PENALTY_BOX_L, pb2),           # 3  big_rect_left_bottom_pt1, 4 pt2
        (0, gb1), (GOAL_BOX_L, gb1),              # 5  small_rect_left_top_pt1, 6 pt2
        (0, gb2), (GOAL_BOX_L, gb2),              # 7  small_rect_left_bottom_pt1, 8 pt2
        (0, W),                                   # 9  sideline_bottom_left
        (arc, W / 2),                             # 10 left_semicircle_right
        (L / 2, 0), (L / 2, W),                   # 11 center_line_top, 12 center_line_bottom
        (L / 2, W / 2 - r), (L / 2, W / 2 + r),   # 13 center_circle_top, 14 center_circle_bottom
        (L / 2, W / 2),                           # 15 field_center
        (L, 0),                                   # 16 sideline_top_right
        (L, pb1), (L - PENALTY_BOX_L, pb1),       # 17 big_rect_right_top_pt1, 18 pt2
        (L, pb2), (L - PENALTY_BOX_L, pb2),       # 19 big_rect_right_bottom_pt1, 20 pt2
        (L, gb1), (L - GOAL_BOX_L, gb1),          # 21 small_rect_right_top_pt1, 22 pt2
        (L, gb2), (L - GOAL_BOX_L, gb2),          # 23 small_rect_right_bottom_pt1, 24 pt2
        (L, W),                                   # 25 sideline_bottom_right
        (L - arc, W / 2),                         # 26 right_semicircle_left
        (L / 2 - r, W / 2), (L / 2 + r, W / 2),   # 27 center_circle_left, 28 center_circle_right
    ], np.float32)


PITCH_VERTICES = pitch_vertices()
N_PITCH_KEYPOINTS = len(PITCH_VERTICES)


def download_pitch_weights() -> str:
    """Baixa o best.pt do Soccana do Hugging Face (fica em cache após a 1ª vez)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Instale huggingface_hub ou informe --pitch-weights com o caminho do .pt")
    print(f"[campo] baixando {PITCH_HF_FILE} de {PITCH_HF_REPO}")
    return hf_hub_download(PITCH_HF_REPO, PITCH_HF_FILE)


class PitchKeypointModel:
    """Detecta os 29 pontos-chave do campo com o modelo Soccana (YOLO11-pose, ultralytics)."""

    def __init__(self, weights: str | None, device: str | None, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("--pitch requer o pacote 'ultralytics': pip install ultralytics")
        path = weights or download_pitch_weights()
        print(f"[campo] carregando {path}")
        self.model = YOLO(path)
        self.device = device
        self.imgsz = imgsz

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devolve (29, 2) posições em px e (29,) confianças; confiança 0 = ponto não encontrado."""
        xy = np.zeros((N_PITCH_KEYPOINTS, 2), np.float32)
        conf = np.zeros(N_PITCH_KEYPOINTS, np.float32)
        kw = {"device": self.device} if self.device else {}
        r = self.model(frame_bgr, imgsz=self.imgsz, verbose=False, **kw)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return xy, conf
        # o modelo detecta o campo como um único objeto; fica com a detecção mais confiável
        k = int(np.argmax(r.boxes.conf.cpu().numpy())) if r.boxes is not None and len(r.boxes) else 0
        pts = r.keypoints.xy[k].cpu().numpy()
        c = r.keypoints.conf[k].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(pts))
        n = min(N_PITCH_KEYPOINTS, len(pts))
        xy[:n], conf[:n] = pts[:n], c[:n]
        return xy, drop_border_keypoints(xy, conf, frame_bgr.shape[1], frame_bgr.shape[0])


def drop_border_keypoints(xy: np.ndarray, conf: np.ndarray, w: int, h: int, margin: float = 3.0) -> np.ndarray:
    """Zera a confiança de pontos colados na borda da imagem.

    O modelo prende na borda os pontos que caem fora do quadro, muitas vezes com
    confiança alta; usá-los distorce a homografia.
    """
    edge = (xy[:, 0] <= margin) | (xy[:, 1] <= margin) | (xy[:, 0] >= w - 1 - margin) | (xy[:, 1] >= h - 1 - margin)
    return np.where(edge, 0.0, conf).astype(np.float32)


# Pontos mal definidos pelo modelo (vértice da meia-lua e tangentes do círculo central)
# pesam menos no ajuste da homografia.
PITCH_KP_WEIGHT = np.ones(N_PITCH_KEYPOINTS, np.float64)
PITCH_KP_WEIGHT[[10, 13, 14, 26, 27, 28]] = 0.5


def project_h(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aplica H a pontos (N, 2); devolve (N, 2) projetados e (N,) componente w (w <= 0 = além do horizonte)."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    w = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / w[:, None], w


def weighted_homography(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Homografia src -> dst por DLT ponderado, com normalização de Hartley."""
    def normalizer(p: np.ndarray) -> np.ndarray:
        c = p.mean(0)
        s = np.sqrt(2) / max(np.linalg.norm(p - c, axis=1).mean(), 1e-9)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = normalizer(src), normalizer(dst)
    a = (np.c_[src, np.ones(len(src))] @ Ts.T)[:, :2]
    b = (np.c_[dst, np.ones(len(dst))] @ Td.T)[:, :2]
    x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.concatenate([np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], 1),
                        np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], 1)])
    A *= np.sqrt(np.tile(weights, 2))[:, None]
    H = np.linalg.inv(Td) @ np.linalg.svd(A)[2][-1].reshape(3, 3) @ Ts
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
        return None
    return H


class CameraMotion:
    """Movimento da câmera entre quadros analisados seguidos, como homografia px -> px.

    A câmera de transmissão gira e dá zoom em torno de um ponto fixo, então o
    movimento de toda a imagem (gramado e arquibancada) é uma homografia. Usa
    fluxo óptico (Lucas-Kanade, ida e volta) fora das caixas dos atletas, que se
    movem por conta própria, e RANSAC para descartar o placar e outros elementos fixos.
    """

    def __init__(self, width: int = 960):
        self.width = width
        self.prev: np.ndarray | None = None
        self.prev_mask: np.ndarray | None = None

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray | None:
        s = min(1.0, self.width / frame.shape[1])
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4) * s:
            pad = 0.15 * (y2 - y1)
            mask[max(0, int(y1 - pad)):int(y2 + pad) + 1, max(0, int(x1 - pad)):int(x2 + pad) + 1] = 0
        M = None
        if self.prev is not None and self.prev.shape == gray.shape:
            M = self._estimate(self.prev, gray, self.prev_mask, s)
        self.prev, self.prev_mask = gray, mask
        return M

    @staticmethod
    def _estimate(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, s: float) -> np.ndarray | None:
        p0 = cv2.goodFeaturesToTrack(prev, 800, 0.01, 8, mask=mask, blockSize=7)
        if p0 is None or len(p0) < 30:
            return None
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1) < 0.7)
        if ok.sum() < 30:
            return None
        Ms, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.0)
        if Ms is None or inl.sum() < max(25, 0.5 * ok.sum()):
            return None
        S = np.diag([s, s, 1.0])
        return np.linalg.inv(S) @ Ms @ S


class PitchMapper:
    """Homografia quadro -> campo (metros), estável entre quadros.

    A cada quadro analisado:
      1. prevê a homografia pelo movimento da câmera (fluxo óptico): H_pred = H_ant · M⁻¹;
      2. ajusta H_obs nos pontos-chave do modelo (RANSAC + DLT ponderado pela confiança);
      3. valida H_obs: erro de reprojeção, horizonte, orientação, área visível e
         coerência com H_pred (salto máximo em metros);
      4. funde H_obs com H_pred (`smooth` = peso da previsão). Sem observação completa,
         corrige a previsão com os pontos visíveis (translação/similaridade na imagem)
         ou segue só com H_pred, por até `flow_max_age` s desde a última observação;
         sem fluxo nem pontos, mantém a última H por até `max_age` s.
    Se a observação for rejeitada por salto `reset_after` vezes seguidas, a previsão
    é considerada perdida e a observação é aceita direto (ressincroniza).
    """

    def __init__(self, model: PitchKeypointModel, kp_conf: float, max_age: float, *,
                 ransac_thr: float = 2.0, max_reproj: float = 2.0, max_jump: float = 3.0,
                 smooth: float = 0.5, flow: bool = True, flow_max_age: float = 10.0,
                 reset_after: int = 3):
        self.model = model
        self.kp_conf = kp_conf
        self.max_age = max_age
        self.ransac_thr = ransac_thr
        self.max_reproj = max_reproj
        self.max_jump = max_jump
        self.smooth = smooth
        self.flow_max_age = flow_max_age
        self.reset_after = reset_after
        self.motion = CameraMotion() if flow else None
        self.H: np.ndarray | None = None
        self.age = float("inf")   # s desde a última observação aceita
        self.hold = 0.0           # s sem nenhuma atualização (nem fluxo)
        self.n_jump_rejects = 0   # observações seguidas rejeitadas por salto
        self.img_size = (1, 1)
        self._ref = np.zeros((0, 2))
        self.fresh = False        # observação aceita neste quadro
        self.source = "none"      # none | obs | fused | partial | flow | hold
        self.rejected = ""        # motivo da rejeição da observação deste quadro
        self.n_inliers = 0
        self.reproj = float("nan")
        self.jump = float("nan")  # distância (m) entre H_obs e a previsão
        self.kp_xy = np.zeros((N_PITCH_KEYPOINTS, 2), np.float32)
        self.kp_mask = np.zeros(N_PITCH_KEYPOINTS, bool)

    @property
    def valid(self) -> bool:
        return self.H is not None

    @property
    def n_points(self) -> int:
        return int(self.kp_mask.sum())

    def update(self, frame: np.ndarray, dt: float, boxes: np.ndarray | None = None) -> None:
        h, w = frame.shape[:2]
        if self.img_size != (w, h):
            self.img_size = (w, h)
            gx, gy = np.meshgrid(np.linspace(0.05, 0.95, 6) * w, np.linspace(0.45, 1.0, 4) * h)
            self._ref = np.stack([gx.ravel(), gy.ravel()], 1)
        M = self.motion(frame, np.zeros((0, 4)) if boxes is None else boxes) if self.motion else None
        H_pred = None
        if self.H is not None and M is not None:
            H_pred = self._normalize(self.H @ np.linalg.inv(M))
            if self._implausible(H_pred):
                H_pred = None

        xy, conf = self.model(frame)
        self.kp_xy, self.kp_mask = xy, conf >= self.kp_conf
        H_obs = self._observe(xy, conf)
        self.jump = float("nan")
        if H_obs is not None and H_pred is not None:
            self.jump = self._distance(H_obs, H_pred)
            if not self.jump <= self.max_jump:
                self.n_jump_rejects += 1
                if self.n_jump_rejects < self.reset_after:
                    H_obs, self.rejected = None, "salto"
                else:
                    H_pred = None  # previsão perdida: ressincroniza na observação

        self.fresh = H_obs is not None
        if H_obs is not None:
            if H_pred is not None and self.smooth > 0:
                self.H, self.source = self._blend(H_pred, H_obs, self.smooth), "fused"
            else:
                self.H, self.source = H_obs, "obs"
            self.age = self.hold = 0.0
            self.n_jump_rejects = 0
        else:
            alive = self.age + dt <= self.flow_max_age
            # previsão pelo fluxo; se o fluxo falhou (movimento brusco), a última H
            base = H_pred if H_pred is not None else self.H
            H_corr = self._correct(base, xy) if base is not None and alive else None
            self.age += dt
            if H_corr is not None:
                if H_pred is not None and self.smooth > 0:
                    H_corr = self._blend(H_pred, H_corr, self.smooth)
                self.H, self.source = H_corr, "partial"
                self.hold = 0.0
            elif H_pred is not None and alive:
                self.H, self.source = H_pred, "flow"
                self.hold = 0.0
            else:
                self.hold += dt
                # com fluxo funcionando mas vencido, a H congelada estaria errada: descarta
                if self.H is not None and H_pred is None and self.hold <= self.max_age:
                    self.source = "hold"
                else:
                    self.H, self.source = None, "none"

    def _correct(self, H_base: np.ndarray, xy: np.ndarray) -> np.ndarray | None:
        """Corrige H_base com os poucos pontos-chave visíveis (menos de 4, ou alinhados).

        Projeta os pontos do campo na imagem por H_base e ajusta uma transformação de
        baixa ordem na imagem até as posições detectadas: translação (1 ponto) ou
        similaridade (2+ pontos: translação, escala e rotação). Cobre a deriva do
        fluxo óptico e as falhas dele em movimentos bruscos da câmera.
        """
        m = self.kp_mask
        if not m.any():
            return None
        pred, _ = project_h(np.linalg.inv(H_base), PITCH_VERTICES[m].astype(np.float64))
        obs = xy[m].astype(np.float64)
        if not np.isfinite(pred).all():
            return None
        tol = 0.01 * self.img_size[0]
        if len(obs) == 1:
            S = np.c_[np.eye(2), obs[0] - pred[0]]
        else:
            method = cv2.RANSAC if len(obs) >= 3 else cv2.LMEDS
            S, _ = cv2.estimateAffinePartial2D(pred, obs, method=method, ransacReprojThreshold=tol)
            if S is None:
                return None
            fit = pred @ S[:, :2].T + S[:, 2]
            if np.median(np.linalg.norm(fit - obs, axis=1)) > tol:
                return None
            scale = np.sqrt(abs(np.linalg.det(S[:, :2])))
            if not 0.67 <= scale <= 1.5 or abs(np.arctan2(S[1, 0], S[0, 0])) > np.radians(10):
                return None
        H = self._normalize(H_base @ np.linalg.inv(np.vstack([S, [0, 0, 1]])))
        return None if self._implausible(H) else H

    def _observe(self, xy: np.ndarray, conf: np.ndarray) -> np.ndarray | None:
        """Ajusta e valida a homografia dos pontos-chave deste quadro; None se não servir."""
        self.rejected, self.n_inliers, self.reproj = "", 0, float("nan")
        m = self.kp_mask
        if m.sum() < 4:
            self.rejected = "poucos pontos" if m.any() else "sem pontos"
            return None
        src = xy[m].astype(np.float64)
        dst = PITCH_VERTICES[m].astype(np.float64)
        wts = conf[m].astype(np.float64) * PITCH_KP_WEIGHT[m]
        # pontos quase alinhados (ex.: só a linha do meio) dão homografia degenerada
        if cv2.contourArea(cv2.convexHull(dst.astype(np.float32))) < 20.0:
            self.rejected = "pontos alinhados"
            return None
        inl = np.ones(len(src), bool)
        if len(src) >= 5:
            H0, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thr)
            if H0 is None:
                self.rejected = "ransac"
                return None
            inl = mask.ravel().astype(bool)
            if inl.sum() < 4:
                self.rejected = "ransac"
                return None
            if cv2.contourArea(cv2.convexHull(dst[inl].astype(np.float32))) < 20.0:
                self.rejected = "pontos alinhados"  # ex.: só a linha do meio sobrou como consenso
                return None
        H = weighted_homography(src[inl], dst[inl], wts[inl])
        if H is None:
            self.rejected = "degenerada"
            return None
        H = self._normalize(H)
        err = np.linalg.norm(project_h(H, src[inl])[0] - dst[inl], axis=1)
        self.n_inliers, self.reproj = int(inl.sum()), float(err.mean())
        if not self.reproj <= self.max_reproj:
            self.rejected = "reprojeção"
            return None
        why = self._implausible(H)
        if why:
            self.rejected = why
            return None
        return H

    def _normalize(self, H: np.ndarray) -> np.ndarray:
        """Escala H para w = 1 no centro da base da imagem (o gramado fica com w > 0)."""
        w, h = self.img_size
        return H / (H[2] @ [w / 2, h, 1.0])

    def _implausible(self, H: np.ndarray) -> str:
        """Motivo para descartar H como fisicamente impossível; '' se plausível."""
        w, h = self.img_size
        quad = np.array([[0, h], [w, h], [w, 0.5 * h], [0, 0.5 * h]], np.float64)
        p, z = project_h(H, quad)
        if not np.isfinite(p).all() or (z <= 0).any():
            return "horizonte"
        # sem espelhamento: imagem e campo têm x para a direita e y para baixo (câmera no lado y = W)
        c = np.array([[w / 2, 0.75 * h], [w / 2 + 1, 0.75 * h], [w / 2, 0.75 * h + 1]])
        q, _ = project_h(H, c)
        a, b = q[1] - q[0], q[2] - q[0]
        if a[0] * b[1] - a[1] * b[0] <= 0:
            return "espelhada"
        area = cv2.contourArea(p.astype(np.float32))
        if not 30.0 <= area <= 4 * PITCH_LENGTH * PITCH_WIDTH:
            return "área visível"
        m = 40.0
        if ((p[:, 0] < -m) | (p[:, 0] > PITCH_LENGTH + m) | (p[:, 1] < -m) | (p[:, 1] > PITCH_WIDTH + m)).any():
            return "fora do campo"
        return ""

    def _ref_pairs(self, Ha: np.ndarray, Hb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pontos de referência do gramado projetados pelas duas homografias (só os válidos em ambas)."""
        pa, za = project_h(Ha, self._ref)
        pb, zb = project_h(Hb, self._ref)
        m = 30.0
        ok = (za > 0) & (zb > 0)
        for p in (pa, pb):
            ok &= (p[:, 0] > -m) & (p[:, 0] < PITCH_LENGTH + m) & (p[:, 1] > -m) & (p[:, 1] < PITCH_WIDTH + m)
        return self._ref[ok], pa[ok], pb[ok]

    def _distance(self, Ha: np.ndarray, Hb: np.ndarray) -> float:
        """Distância média (m) entre as projeções do gramado visível pelas duas homografias."""
        _, pa, pb = self._ref_pairs(Ha, Hb)
        return float(np.linalg.norm(pa - pb, axis=1).mean()) if len(pa) >= 4 else float("inf")

    def _blend(self, H_pred: np.ndarray, H_obs: np.ndarray, w_pred: float) -> np.ndarray:
        """Média ponderada das duas homografias, feita no espaço do campo sobre pontos de referência."""
        img, pa, pb = self._ref_pairs(H_pred, H_obs)
        if len(img) < 6:
            return H_obs
        H, _ = cv2.findHomography(img, w_pred * pa + (1 - w_pred) * pb, 0)
        if H is None or not np.isfinite(H).all():
            return H_obs
        return self._normalize(H)

    def to_pitch(self, pts: np.ndarray) -> np.ndarray | None:
        """Converte pontos (N, 2) em px para metros no campo; None sem homografia."""
        if self.H is None:
            return None
        if len(pts) == 0:
            return np.zeros((0, 2), np.float32)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.H).reshape(-1, 2)


def anchor_points(det: Detections, names: list[str]) -> np.ndarray:
    """Ponto de cada detecção que toca o gramado: pés (base da caixa) para pessoas, centro para a bola.

    A bola no alto é projetada no ponto do gramado atrás dela, não sob ela.
    """
    x1, y1, x2, y2 = det.xyxy.T
    is_ball = np.array([class_name(names, int(c)) == "ball" for c in det.class_id], bool)
    return np.stack([(x1 + x2) / 2, np.where(is_ball, (y1 + y2) / 2, y2)], axis=1).astype(np.float32)


def pitch_lines() -> list[np.ndarray]:
    """Linhas do campo de referência (m) como polilinhas densas, para desenhar na imagem."""
    L, W = PITCH_LENGTH, PITCH_WIDTH

    def seg(a, b, n=24):
        return np.linspace(a, b, n)

    def poly(*pts):
        return np.concatenate([seg(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])

    def arc(c, r, t0, t1, n=48):
        t = np.linspace(t0, t1, n)
        return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)

    pb1, pb2 = (W - PENALTY_BOX_W) / 2, (W + PENALTY_BOX_W) / 2
    gb1, gb2 = (W - GOAL_BOX_W) / 2, (W + GOAL_BOX_W) / 2
    t = np.arccos((PENALTY_BOX_L - PENALTY_SPOT) / CENTRE_CIRCLE_R)  # meia-lua fora da área
    return [
        poly((0, 0), (L, 0), (L, W), (0, W), (0, 0)),
        seg((L / 2, 0), (L / 2, W)),
        arc((L / 2, W / 2), CENTRE_CIRCLE_R, 0, 2 * np.pi, 96),
        poly((0, pb1), (PENALTY_BOX_L, pb1), (PENALTY_BOX_L, pb2), (0, pb2)),
        poly((L, pb1), (L - PENALTY_BOX_L, pb1), (L - PENALTY_BOX_L, pb2), (L, pb2)),
        poly((0, gb1), (GOAL_BOX_L, gb1), (GOAL_BOX_L, gb2), (0, gb2)),
        poly((L, gb1), (L - GOAL_BOX_L, gb1), (L - GOAL_BOX_L, gb2), (L, gb2)),
        arc((PENALTY_SPOT, W / 2), CENTRE_CIRCLE_R, -t, t),
        arc((L - PENALTY_SPOT, W / 2), CENTRE_CIRCLE_R, np.pi - t, np.pi + t),
    ]


PITCH_LINES = pitch_lines()
# Cor (BGR) das linhas projetadas conforme a origem da homografia no quadro
PITCH_SOURCE_COLORS = {"obs": (0, 255, 0), "fused": (0, 255, 0), "partial": (255, 255, 0),
                       "flow": (0, 220, 255), "hold": (0, 0, 255)}


def draw_pitch_keypoints(img: np.ndarray, mapper: PitchMapper) -> None:
    """Marca os pontos-chave do campo e desenha as linhas do campo projetadas pela homografia.

    Linhas verdes = homografia observada neste quadro, ciano = fluxo corrigido por
    poucos pontos, amarelas = só fluxo óptico, vermelhas = homografia antiga mantida. Se não coincidirem com as linhas do
    gramado, a homografia está errada.
    """
    if mapper.H is not None:
        Hi = np.linalg.inv(mapper.H)
        color = PITCH_SOURCE_COLORS.get(mapper.source, (255, 255, 255))
        th = max(1, img.shape[0] // 540)
        for line in PITCH_LINES:
            p, z = project_h(Hi, line)
            ok = (z > 0) & np.isfinite(p).all(1) & (np.abs(p) < 1e5).all(1)
            # quebra a polilinha onde há pontos inválidos (além do horizonte)
            for run in np.split(np.arange(len(p)), np.flatnonzero(~ok)):
                run = run[ok[run]]
                if len(run) >= 2:
                    cv2.polylines(img, [p[run].round().astype(np.int32)], False, color, th, cv2.LINE_AA)
        cv2.putText(img, f"homografia: {mapper.source}" + (f" ({mapper.rejected})" if mapper.rejected else ""),
                    (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[0] / 1800),
                    color, max(1, th), cv2.LINE_AA)
    r = max(3, img.shape[0] // 270)
    for i in np.flatnonzero(mapper.kp_mask):
        x, y = map(int, mapper.kp_xy[i])
        cv2.circle(img, (x, y), r, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (x + r + 2, y - r), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, img.shape[0] / 2700), (255, 0, 255), 1, cv2.LINE_AA)


def draw_radar(img: np.ndarray, pts: np.ndarray | None, det: Detections, names: list[str],
               teams: np.ndarray, team_colors: list | None, holder: int | None) -> None:
    """Painel no canto inferior direito com a posição dos atletas no campo (vista de cima).

    `pts` são as posições em metros de cada detecção (None = campo não identificado;
    NaN = posição removida pelo filtro de velocidade).

    Jogadores: círculo na cor da equipe (contorno vermelho = jogador com a bola).
    Goleiros: quadrado na cor da equipe. Árbitros: círculo preto. Bola: círculo amarelo.
    """
    H, W = img.shape[:2]
    s = H / 1080
    font = cv2.FONT_HERSHEY_SIMPLEX
    t1, t2 = max(1, int(2 * s)), max(1, int(1.5 * s))
    m, pw = int(20 * s), int(480 * s)
    inner = int(14 * s)                      # margem interna do painel
    title_h, legend_h = int(44 * s), int(34 * s)
    pad = 3.0                                # metros de gramado fora das linhas
    fw = pw - 2 * inner                      # largura do desenho do campo
    scale = fw / (PITCH_LENGTH + 2 * pad)
    fh = int((PITCH_WIDTH + 2 * pad) * scale)
    ph = title_h + fh + legend_h + inner
    x0, y0 = W - m - pw, H - m - ph
    if x0 < 0 or y0 < 0:
        return
    roi = img[y0:y0 + ph, x0:x0 + pw]
    roi[:] = cv2.addWeighted(roi, 0.35, np.full_like(roi, 20), 0.65, 0)
    cv2.putText(img, "POSICIONAMENTO", (x0 + inner, y0 + int(32 * s)), font, 0.75 * s,
                (255, 255, 255), t1, cv2.LINE_AA)

    fx0, fy0 = x0 + inner, y0 + title_h
    field = img[fy0:fy0 + fh, fx0:fx0 + fw]
    field[:] = cv2.addWeighted(field, 0.25, np.full_like(field, (45, 120, 45)), 0.75, 0)

    def px(p) -> tuple[int, int]:
        return int(fx0 + (p[0] + pad) * scale), int(fy0 + (p[1] + pad) * scale)

    line = (235, 235, 235)
    L, Wd = PITCH_LENGTH, PITCH_WIDTH

    def rect(xa, ya, xb, yb) -> None:
        cv2.rectangle(img, px((xa, ya)), px((xb, yb)), line, 1, cv2.LINE_AA)

    rect(0, 0, L, Wd)
    cv2.line(img, px((L / 2, 0)), px((L / 2, Wd)), line, 1, cv2.LINE_AA)
    for bw, bl in ((PENALTY_BOX_W, PENALTY_BOX_L), (GOAL_BOX_W, GOAL_BOX_L)):
        rect(0, (Wd - bw) / 2, bl, (Wd + bw) / 2)
        rect(L - bl, (Wd - bw) / 2, L, (Wd + bw) / 2)
    rr = int(round(CENTRE_CIRCLE_R * scale))
    cv2.circle(img, px((L / 2, Wd / 2)), rr, line, 1, cv2.LINE_AA)
    # meia-lua: parte do círculo da marca do pênalti que fica fora da área
    half = float(np.degrees(np.arccos((PENALTY_BOX_L - PENALTY_SPOT) / CENTRE_CIRCLE_R)))
    for spot, mid in ((PENALTY_SPOT, 0.0), (L - PENALTY_SPOT, 180.0)):
        c = px((spot, Wd / 2))
        cv2.ellipse(img, c, (rr, rr), 0, mid - half, mid + half, line, 1, cv2.LINE_AA)
        cv2.circle(img, c, max(1, int(1.5 * s)), line, -1, cv2.LINE_AA)

    r = max(3, int(6 * s))
    ref_color = CLASS_COLORS["referee"]
    ball_color = CLASS_COLORS["ball"]

    def marker(c, kind: str, color, ring=(0, 0, 0), ring_w=1) -> None:
        if kind == "goalkeeper":
            cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), color, -1)
            cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), ring, ring_w)
        elif kind == "ball":
            cv2.circle(img, c, max(2, r - 2), color, -1, cv2.LINE_AA)
            cv2.circle(img, c, max(2, r - 2), (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, ring, ring_w, cv2.LINE_AA)

    # legenda
    ly = y0 + title_h + fh + int(24 * s)
    items = []
    if team_colors:
        items += [("player", team_colors[0], TEAM_NAMES[0]), ("player", team_colors[1], TEAM_NAMES[1])]
    else:
        items += [("player", CLASS_COLORS["player"], "Jogador")]
    items += [("referee", ref_color, "Arbitro"), ("ball", ball_color, "Bola")]
    lx = x0 + inner + r
    for kind, color, label in items:
        marker((lx, ly - int(6 * s)), kind, color, (255, 255, 255) if kind == "referee" else (0, 0, 0))
        cv2.putText(img, label, (lx + r + int(6 * s), ly), font, 0.5 * s, (255, 255, 255), t2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, font, 0.5 * s, t2)
        lx += r + int(6 * s) + tw + int(22 * s) + r

    if pts is None:
        cv2.putText(img, "campo nao identificado", (fx0 + int(10 * s), fy0 + int(26 * s)), font,
                    0.6 * s, (255, 255, 255), t2, cv2.LINE_AA)
        return
    order = sorted(range(len(det)), key=lambda i: class_name(names, int(det.class_id[i])) == "ball")
    for i in order:  # bola por último, por cima dos atletas
        x, y = pts[i]
        if not (-pad <= x <= PITCH_LENGTH + pad and -pad <= y <= PITCH_WIDTH + pad):
            continue
        name = class_name(names, int(det.class_id[i]))
        team = int(teams[i]) if i < len(teams) else -1
        color = team_colors[team] if team >= 0 and team_colors else CLASS_COLORS.get(name, FALLBACK_COLOR)
        ring, ring_w = ((0, 0, 255), 2) if i == holder else \
            (((255, 255, 255), 1) if name == "referee" else ((0, 0, 0), 1))
        marker(px((x, y)), name, color, ring, ring_w)


# --------------------------------------------------------------------------- #
# Filtro de velocidade das posições no campo
# --------------------------------------------------------------------------- #
PERSON_CLASSES = ("player", "goalkeeper", "referee")
POS_OK, POS_PENDING, POS_INTERP, POS_REMOVED = 0, 1, 2, 3
POS_FLAG_LABELS = {POS_OK: "", POS_PENDING: "", POS_INTERP: "interpolado", POS_REMOVED: "removido"}


class FrameRecord:
    """Dados de um quadro analisado, guardados até as posições no campo ficarem definitivas."""

    def __init__(self, idx: int, ts: float, det: Detections, teams: np.ndarray,
                 holder: int | None, raw: np.ndarray | None):
        self.idx, self.ts, self.det, self.teams, self.holder = idx, ts, det, teams, holder
        self.raw = raw                                   # (N, 2) metros, como saiu da homografia
        self.pts = None if raw is None else raw.copy()   # (N, 2) após o filtro; NaN = removida
        self.flag = np.zeros(len(det), int)
        self.track = np.full(len(det), -1, int)
        self.pending = 0                                 # posições ainda sem decisão
        self.imgs: list[np.ndarray] = []                 # quadros de vídeo à espera deste registro
        self.released = False


class _Track:
    def __init__(self, tid: int, box: np.ndarray, ts: float):
        self.id, self.box, self.seen = tid, box, ts
        self.anchor: np.ndarray | None = None  # última posição válida (metros)
        self.anchor_t = 0.0
        self.pending: list[tuple[FrameRecord, int]] = []


class PitchSpeedFilter:
    """Remove saltos impossíveis na posição de cada atleta no campo e interpola o buraco.

    Os atletas são ligados de um quadro ao outro pela caixa na imagem (distância entre
    os pés, em alturas de caixa), não pela posição em metros, que é justamente o que pode
    estar errado. Para cada atleta, se a distância até a última posição válida dividida
    pelo tempo passado passar de `max_speed` (m/s), a posição fica pendente; quando chega
    a próxima posição válida, as pendentes viram a interpolação linear entre as duas.
    Se nenhuma posição válida chegar em `max_gap` segundos, as pendentes são removidas;
    se as duas últimas pendentes forem coerentes entre si, a mais recente passa a ser a
    nova referência (a referência antiga é que estava errada).
    """

    def __init__(self, names: list[str], max_speed: float, max_gap: float, match_dist: float):
        self.names = names
        self.max_speed, self.max_gap, self.match_dist = max_speed, max_gap, match_dist
        self.tracks: list[_Track] = []
        self.next_id = 0
        self.n_checked = self.n_interp = self.n_removed = 0

    def update(self, rec: FrameRecord) -> None:
        people = [k for k, c in enumerate(rec.det.class_id)
                  if class_name(self.names, int(c)) in PERSON_CLASSES]
        self._associate(rec, people)
        if rec.pts is not None:
            by_id = {tr.id: tr for tr in self.tracks}
            for k in people:
                self._check(by_id[int(rec.track[k])], rec, k)
        self._expire(rec.ts)

    def flush(self) -> None:
        """Fim do vídeo: pendentes sem próxima posição válida são removidas."""
        for tr in self.tracks:
            for r, j in tr.pending:
                self._finish(r, j, POS_REMOVED, np.nan)
            tr.pending = []

    def _associate(self, rec: FrameRecord, people: list[int]) -> None:
        boxes = rec.det.xyxy[people]
        cands = []
        for ti, tr in enumerate(self.tracks):
            for j, b in enumerate(boxes):
                h = ((tr.box[3] - tr.box[1]) + (b[3] - b[1])) / 2
                d = float(np.hypot((tr.box[0] + tr.box[2] - b[0] - b[2]) / 2, tr.box[3] - b[3]))
                if h > 0 and d <= self.match_dist * h:
                    cands.append((d / h, ti, j))
        cands.sort()
        used_t, used_d = set(), set()
        for _, ti, j in cands:
            if ti in used_t or j in used_d:
                continue
            used_t.add(ti)
            used_d.add(j)
            self._assign(self.tracks[ti], rec, people[j])
        for j, k in enumerate(people):
            if j not in used_d:
                tr = _Track(self.next_id, rec.det.xyxy[k], rec.ts)
                self.next_id += 1
                self.tracks.append(tr)
                self._assign(tr, rec, k)

    @staticmethod
    def _assign(tr: _Track, rec: FrameRecord, k: int) -> None:
        tr.box, tr.seen = rec.det.xyxy[k].copy(), rec.ts
        rec.track[k] = tr.id

    def _fast(self, a: np.ndarray, ta: float, b: np.ndarray, tb: float) -> bool:
        return float(np.linalg.norm(b - a)) > self.max_speed * (tb - ta)

    def _check(self, tr: _Track, rec: FrameRecord, k: int) -> None:
        p, t = rec.raw[k], rec.ts
        self.n_checked += 1
        if not np.isfinite(p).all():
            rec.pending += 1
            self._finish(rec, k, POS_REMOVED, np.nan)
            return
        if tr.anchor is None or not self._fast(tr.anchor, tr.anchor_t, p, t):
            for r, j in tr.pending:  # buraco fechado: interpola entre as duas posições válidas
                a = (r.ts - tr.anchor_t) / (t - tr.anchor_t)
                self._finish(r, j, POS_INTERP, tr.anchor + a * (p - tr.anchor))
            tr.pending = []
            tr.anchor, tr.anchor_t = p.copy(), t
        else:
            rec.flag[k] = POS_PENDING
            rec.pending += 1
            tr.pending.append((rec, k))

    def _expire(self, now: float) -> None:
        keep = []
        for tr in self.tracks:
            lost = now - tr.seen > self.max_gap
            if tr.pending and (lost or now - tr.pending[0][0].ts > self.max_gap):
                newest = None
                if not lost and len(tr.pending) >= 2:
                    (r1, j1), (r2, j2) = tr.pending[-2:]
                    if not self._fast(r1.raw[j1], r1.ts, r2.raw[j2], r2.ts):
                        newest = tr.pending.pop()
                for r, j in tr.pending:
                    self._finish(r, j, POS_REMOVED, np.nan)
                tr.pending = []
                if newest:
                    r, j = newest
                    self._finish(r, j, POS_OK, r.raw[j])
                    tr.anchor, tr.anchor_t = r.raw[j].copy(), r.ts
            if not lost:
                keep.append(tr)
        self.tracks = keep

    def _finish(self, rec: FrameRecord, k: int, flag: int, value) -> None:
        rec.pts[k] = value
        rec.flag[k] = flag
        rec.pending -= 1
        if flag == POS_INTERP:
            self.n_interp += 1
        elif flag == POS_REMOVED:
            self.n_removed += 1


# --------------------------------------------------------------------------- #
# Escrita do vídeo
# --------------------------------------------------------------------------- #
class VideoSink:
    """Grava H.264 via ffmpeg (toca em qualquer navegador); cai para OpenCV se não houver ffmpeg."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.size = size
        w, h = size
        self.proc = None
        self.writer = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def fit(self, frame: np.ndarray) -> np.ndarray:
        """Redimensiona para o tamanho de saída (para guardar quadros pequenos no buffer)."""
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return frame

    def write(self, frame: np.ndarray) -> None:
        frame = self.fit(frame)
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.writer:
            self.writer.release()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["frame", "timestamp", "class_id", "class_name", "confidence",
              "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y", "area", "team",
              "pitch_x", "pitch_y", "pitch_flag", "pitch_x_raw", "pitch_y_raw", "track_id"]
POSSESSION_FIELDS = ["frame", "timestamp", "ball_detected", "team_near_ball", "possession_team",
                     "seconds_A", "seconds_B", "pct_A", "pct_B"]
PITCH_FIELDS = ["frame", "timestamp", "keypoints", "homography", "fresh",
                "source", "inliers", "reproj_m", "jump_m", "rejected",
                *[f"h{r}{c}" for r in (1, 2, 3) for c in (1, 2, 3)]]


def parse_class_thresholds(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, val = item.partition("=")
        out[name.strip().lower()] = float(val)
    return out


def apply_class_thresholds(det: Detections, names: list[str], per_class: dict[str, float]) -> Detections:
    if not per_class or len(det) == 0:
        return det
    keep = np.array([det.confidence[i] >= per_class.get(class_name(names, int(c)), 0.0)
                     for i, c in enumerate(det.class_id)])
    return det.filter(keep)


def keep_best_ball(det: Detections, names: list[str]) -> Detections:
    """Só existe uma bola em campo: mantém apenas a de maior confiança."""
    is_ball = np.array([class_name(names, int(c)) == "ball" for c in det.class_id], bool)
    if is_ball.sum() <= 1:
        return det
    best = np.flatnonzero(is_ball)[np.argmax(det.confidence[is_ball])]
    keep = ~is_ball
    keep[best] = True
    return det.filter(keep)


def run(args: argparse.Namespace, detector=None) -> dict:
    src = Path(args.video)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir o vídeo: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = int(args.start * fps)
    end = total if args.end is None else min(total, int(args.end * fps))
    if args.max_frames:
        end = min(end, start + args.max_frames)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    print(f"[video] {src.name}: {W}x{H} @ {fps:.2f} fps, {total} quadros — processando {start}..{end - 1}")

    if detector is None:
        names = [n.strip().lower() for n in args.class_names.split(",")] if args.class_names else None
        detector = SoccerDetector(args.weights, args.device, args.resolution, names, args.optimize)
    names = detector.class_names
    per_class = parse_class_thresholds(args.class_threshold)
    base_thr = min([args.threshold, *per_class.values()]) if per_class else args.threshold

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    csv_path = out_dir / f"{stem}_detections.csv"
    video_path = out_dir / f"{stem}_annotated.mp4"

    def detect(frame: np.ndarray) -> Detections:
        det = (detect_tiled(detector, frame, base_thr, args.tiles, args.tile_overlap)
               if args.tiles > 1 else detector(frame, base_thr))
        det = det.filter(det.confidence >= args.threshold) if not per_class else \
            apply_class_thresholds(det, names, {n: per_class.get(n, args.threshold) for n in names})
        return keep_best_ball(det, names) if args.single_ball else det

    teams_clf, tracker, poss_path = None, None, None
    if not args.no_possession:
        print(f"[posse] calibrando cores das equipes em {args.team_calib_frames} quadros...")
        teams_clf = calibrate_teams(cap, detect, start, end, args.team_calib_frames, names)
        if teams_clf.fitted:
            print(f"[posse] cores (BGR): A={teams_clf.colors[0]}  B={teams_clf.colors[1]}")
            tracker = PossessionTracker(args.possession_distance * H / 1080, args.possession_min_seconds)
            poss_path = out_dir / f"{stem}_possession.csv"
        else:
            print("[posse] jogadores insuficientes para separar as equipes; posse desativada")
            teams_clf = None

    mapper, pitch_path = None, None
    if args.pitch:
        kp_model = PitchKeypointModel(args.pitch_weights, args.device, args.pitch_imgsz)
        mapper = PitchMapper(kp_model, args.pitch_kp_conf, args.pitch_max_age,
                             ransac_thr=args.pitch_ransac_thr, max_reproj=args.pitch_max_reproj,
                             max_jump=args.pitch_max_jump, smooth=args.pitch_smooth,
                             flow=not args.no_pitch_flow, flow_max_age=args.pitch_flow_max_age)
        pitch_path = out_dir / f"{stem}_pitch.csv"
    speed_filter = None
    if mapper and args.max_speed > 0:
        speed_filter = PitchSpeedFilter(names, args.max_speed, args.max_gap, args.track_match_dist)

    sink = None
    if not args.no_video:
        out_w = min(args.output_width or W, W)
        out_h = int(round(H * out_w / W / 2) * 2)
        sink = VideoSink(video_path, fps / (1 if args.reuse_detections else args.frame_skip),
                         (out_w, out_h))

    counts: Counter[str] = Counter()
    frames_done = 0
    infer_time = 0.0
    last = Detections.empty()
    last_teams = np.zeros(0, int)
    queue: deque[FrameRecord] = deque()  # quadros à espera do filtro de velocidade
    last_rec: FrameRecord | None = None
    t0 = time.time()
    dt = args.frame_skip / fps  # segundos de vídeo representados por quadro analisado

    poss_fh = open(poss_path, "w", newline="") if poss_path else None
    poss_writer = csv.DictWriter(poss_fh, fieldnames=POSSESSION_FIELDS) if poss_fh else None
    if poss_writer:
        poss_writer.writeheader()
    pitch_fh = open(pitch_path, "w", newline="") if pitch_path else None
    pitch_writer = csv.DictWriter(pitch_fh, fieldnames=PITCH_FIELDS) if pitch_fh else None
    if pitch_writer:
        pitch_writer.writeheader()
    pitch_frames_ok = 0
    pitch_sources: Counter[str] = Counter()

    def fmt(v) -> float | str:
        return round(float(v), 2) if np.isfinite(v) else ""

    def write_rows(rec: FrameRecord) -> None:
        det = rec.det
        for k, ((x1, y1, x2, y2), conf, cid) in enumerate(zip(det.xyxy, det.confidence, det.class_id)):
            name = class_name(names, int(cid))
            counts[name] += 1
            w, h = x2 - x1, y2 - y1
            has = rec.pts is not None
            writer.writerow({
                "frame": rec.idx, "timestamp": round(rec.ts, 3), "class_id": int(cid),
                "class_name": name, "confidence": round(float(conf), 4),
                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                "width": round(float(w), 1), "height": round(float(h), 1),
                "center_x": round(float(x1 + w / 2), 1), "center_y": round(float(y1 + h / 2), 1),
                "area": round(float(w * h), 1),
                "team": TEAM_SHORT[rec.teams[k]] if rec.teams[k] >= 0 else "",
                "pitch_x": fmt(rec.pts[k, 0]) if has else "",
                "pitch_y": fmt(rec.pts[k, 1]) if has else "",
                "pitch_flag": POS_FLAG_LABELS[int(rec.flag[k])] if has else "",
                "pitch_x_raw": fmt(rec.raw[k, 0]) if has else "",
                "pitch_y_raw": fmt(rec.raw[k, 1]) if has else "",
                "track_id": int(rec.track[k]) if rec.track[k] >= 0 else "",
            })

    def finish_img(rec: FrameRecord, img: np.ndarray) -> None:
        if mapper and not args.no_radar:
            draw_radar(img, rec.pts, rec.det, names, rec.teams,
                       teams_clf.colors if teams_clf else None, rec.holder)
        sink.write(img)

    def release() -> None:
        """Grava, em ordem, os quadros cujas posições no campo já estão decididas."""
        while queue and queue[0].pending == 0:
            rec = queue.popleft()
            write_rows(rec)
            for img in rec.imgs:
                finish_img(rec, img)
            rec.imgs = []
            rec.released = True

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            run_model = (idx - start) % args.frame_skip == 0
            if run_model:
                ti = time.time()
                det = detect(frame)
                infer_time += time.time() - ti
                last = det
                last_teams = teams_clf.predict(frame, det, names) if teams_clf else np.full(len(det), -1)
                frames_done += 1
                ts = idx / fps
                pitch_pts = None
                if mapper:
                    mapper.update(frame, dt, det.xyxy)
                    pitch_pts = mapper.to_pitch(anchor_points(det, names))
                    pitch_frames_ok += mapper.valid
                    pitch_sources[mapper.source] += 1
                    Hm = [float(v) for v in mapper.H.ravel()] if mapper.valid else [""] * 9
                    pitch_writer.writerow({
                        "frame": idx, "timestamp": round(ts, 3), "keypoints": mapper.n_points,
                        "homography": int(mapper.valid), "fresh": int(mapper.fresh),
                        "source": mapper.source, "inliers": mapper.n_inliers,
                        "reproj_m": fmt(mapper.reproj), "jump_m": fmt(mapper.jump),
                        "rejected": mapper.rejected,
                        **dict(zip(PITCH_FIELDS[10:], Hm)),
                    })
                if tracker:
                    near = tracker.update(det, last_teams, names, dt)
                    pct = tracker.percentages()
                    poss_writer.writerow({
                        "frame": idx, "timestamp": round(ts, 3),
                        "ball_detected": int(any(class_name(names, int(c)) == "ball" for c in det.class_id)),
                        "team_near_ball": TEAM_SHORT[near] if near >= 0 else "",
                        "possession_team": TEAM_SHORT[tracker.team] if tracker.team >= 0 else "",
                        "seconds_A": round(tracker.seconds[0], 3), "seconds_B": round(tracker.seconds[1], 3),
                        "pct_A": round(pct[0], 2), "pct_B": round(pct[1], 2),
                    })
                last_rec = FrameRecord(idx, ts, det, last_teams, tracker.holder if tracker else None,
                                       pitch_pts)
                if speed_filter:
                    speed_filter.update(last_rec)
                queue.append(last_rec)
            if sink and (run_model or args.reuse_detections):
                if tracker:
                    img = annotate(frame, last, names, not args.hide_conf, last_teams,
                                   teams_clf.colors, last_rec.holder)
                    draw_possession_panel(img, tracker, teams_clf.colors)
                else:
                    img = annotate(frame, last, names, not args.hide_conf)
                if mapper and args.pitch_show_keypoints:
                    draw_pitch_keypoints(img, mapper)
                img = sink.fit(img)  # o minimapa é desenhado quando as posições ficam definitivas
                if last_rec.released:
                    finish_img(last_rec, img)
                else:
                    last_rec.imgs.append(img)
            release()
            if run_model and frames_done % 10 == 0:
                el = time.time() - t0
                n_left = (end - idx - 1) / args.frame_skip
                print(f"  quadro {idx}/{end - 1} | {frames_done / el:.2f} q/s | "
                      f"restam ~{n_left * el / frames_done:.0f}s", flush=True)
        if speed_filter:
            speed_filter.flush()
        release()

    cap.release()
    if sink:
        sink.close()
    if poss_fh:
        poss_fh.close()
    if pitch_fh:
        pitch_fh.close()

    elapsed = time.time() - t0
    print(f"\n[ok] {frames_done} quadros analisados em {elapsed:.1f}s "
          f"(inferência média {1000 * infer_time / max(frames_done, 1):.0f} ms/quadro)")
    print(f"[ok] CSV:   {csv_path}")
    if sink:
        print(f"[ok] vídeo: {video_path}")
    print("\nDetecções por classe (média por quadro analisado):")
    for name in names:
        if counts[name] or name in DEFAULT_CLASS_NAMES:
            print(f"  {LABELS_PT.get(name, name):10s} {counts[name]:6d}  ({counts[name] / max(frames_done, 1):.1f}/quadro)")
    possession = None
    if tracker:
        pct = tracker.percentages()
        print(f"\nPosse de bola (CSV: {poss_path}):")
        for t in (0, 1):
            print(f"  {TEAM_NAMES[t]}  {fmt_time(tracker.seconds[t])}  {pct[t]:5.1f}%")
        possession = {"seconds": list(tracker.seconds), "pct": pct, "csv": str(poss_path)}
    pitch = None
    if mapper:
        ratio = pitch_frames_ok / max(frames_done, 1)
        print(f"\nHomografia do campo (CSV: {pitch_path}): {pitch_frames_ok}/{frames_done} "
              f"quadros ({100 * ratio:.0f}%)")
        src_counts = dict(sorted(pitch_sources.items()))
        print("  origem: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))
        pitch = {"frames_ok": pitch_frames_ok, "ratio": ratio, "csv": str(pitch_path), "sources": src_counts}
        if speed_filter:
            sf = speed_filter
            print(f"Filtro de velocidade (> {args.max_speed:g} m/s): {sf.n_checked} posições de atletas, "
                  f"{sf.n_interp} interpoladas, {sf.n_removed} removidas")
            pitch.update(speed_checked=sf.n_checked, speed_interpolated=sf.n_interp,
                         speed_removed=sf.n_removed)
    return {"frames": frames_done, "counts": dict(counts), "csv": str(csv_path),
            "video": str(video_path) if sink else None, "possession": possession, "pitch": pitch}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detecta bola, jogadores, árbitros e goleiros em vídeo de futebol (RF-DETR SoccerNet).")
    p.add_argument("video", help="arquivo de vídeo de entrada")
    p.add_argument("-o", "--output-dir", default="saida", help="pasta de saída (padrão: ./saida)")
    p.add_argument("--weights", help="checkpoint .pth local (padrão: baixa do Hugging Face)")
    p.add_argument("--device", help="cpu, cuda, cuda:0 ou mps (padrão: automático)")
    p.add_argument("--resolution", type=int, default=None,
                   help="resolução de entrada do modelo (padrão: a do checkpoint; o Large legado exige múltiplo de 56, ex.: 1288)")
    p.add_argument("--threshold", type=float, default=0.5, help="confiança mínima (padrão 0.5)")
    p.add_argument("--class-threshold", action="append", default=[], metavar="CLASSE=VALOR",
                   help="limiar por classe, ex.: --class-threshold player=0.7 --class-threshold ball=0.4")
    p.add_argument("--class-names", help="nomes das classes na ordem dos IDs, separados por vírgula "
                                         f"(padrão: do checkpoint, ou {','.join(DEFAULT_CLASS_NAMES)})")
    p.add_argument("--frame-skip", type=int, default=1, help="analisa 1 a cada N quadros (padrão 1)")
    p.add_argument("--reuse-detections", action="store_true",
                   help="com --frame-skip, grava todos os quadros no vídeo repetindo a última detecção")
    p.add_argument("--start", type=float, default=0.0, help="início em segundos")
    p.add_argument("--end", type=float, default=None, help="fim em segundos")
    p.add_argument("--max-frames", type=int, default=None, help="limite de quadros a ler")
    p.add_argument("--tiles", type=int, default=1,
                   help="divide o quadro em N x N blocos sobrepostos (ex.: 2 para 4K); melhora bola/jogadores pequenos, custa N²+1 inferências")
    p.add_argument("--tile-overlap", type=float, default=0.2, help="sobreposição entre blocos (padrão 0.2)")
    p.add_argument("--single-ball", action="store_true", help="mantém só a bola de maior confiança por quadro")
    p.add_argument("--output-width", type=int, default=1920, help="largura do vídeo anotado (padrão 1920; 0 = original)")
    p.add_argument("--no-video", action="store_true", help="não gera o vídeo anotado, só o CSV")
    p.add_argument("--hide-conf", action="store_true", help="não mostra a confiança nos rótulos")
    p.add_argument("--optimize", action="store_true", help="usa optimize_for_inference() (mais rápido em GPU)")
    p.add_argument("--no-possession", action="store_true", help="desativa equipes e posse de bola")
    p.add_argument("--possession-distance", type=float, default=100,
                   help="distância máx. bola-jogador para posse, em px de um quadro 1080p (escala com a resolução; padrão 100)")
    p.add_argument("--possession-min-seconds", type=float, default=3.0,
                   help="segundos seguidos com a outra equipe para a posse trocar (padrão 3)")
    p.add_argument("--team-calib-frames", type=int, default=10,
                   help="quadros amostrados para aprender as cores das equipes (padrão 10)")
    p.add_argument("--pitch", action="store_true",
                   help="detecta os pontos do campo (modelo Soccana) e calcula a homografia para metros")
    p.add_argument("--pitch-weights",
                   help=f"pesos .pt locais do modelo de campo (padrão: baixa {PITCH_HF_FILE} de {PITCH_HF_REPO})")
    p.add_argument("--pitch-imgsz", type=int, default=640,
                   help="resolução de entrada do modelo de campo (padrão 640, a do treino)")
    p.add_argument("--pitch-kp-conf", type=float, default=0.5,
                   help="confiança mínima de cada ponto do campo (padrão 0.5)")
    p.add_argument("--pitch-max-age", type=float, default=0.5,
                   help="segundos que a última homografia vale quando faltam pontos e o fluxo óptico falha (padrão 0.5)")
    p.add_argument("--pitch-ransac-thr", type=float, default=2.0,
                   help="tolerância do RANSAC dos pontos do campo, em metros (padrão 2.0)")
    p.add_argument("--pitch-max-reproj", type=float, default=2.0,
                   help="erro médio de reprojeção máx. dos pontos usados, em metros (padrão 2.0)")
    p.add_argument("--pitch-max-jump", type=float, default=3.0,
                   help="diferença máx. (m) entre a homografia observada e a prevista pelo movimento "
                        "da câmera; acima disso a observação é descartada (padrão 3)")
    p.add_argument("--pitch-smooth", type=float, default=0.5,
                   help="peso da previsão pelo fluxo óptico na fusão com a observação "
                        "(0 = só observação, padrão 0.5)")
    p.add_argument("--pitch-flow-max-age", type=float, default=10.0,
                   help="segundos que a homografia segue só pelo fluxo óptico sem observação válida (padrão 10)")
    p.add_argument("--no-pitch-flow", action="store_true",
                   help="não estima o movimento da câmera por fluxo óptico")
    p.add_argument("--pitch-show-keypoints", action="store_true",
                   help="desenha no vídeo os pontos do campo detectados")
    p.add_argument("--no-radar", action="store_true", help="não desenha o minimapa do campo")
    p.add_argument("--max-speed", type=float, default=11.0,
                   help="velocidade máx. de um atleta no campo, em m/s; posições acima disso são "
                        "removidas e interpoladas até a próxima válida (padrão 11; 0 desativa)")
    p.add_argument("--max-gap", type=float, default=1.0,
                   help="segundos máx. esperando a próxima posição válida para interpolar; "
                        "depois disso as posições são removidas (padrão 1)")
    p.add_argument("--track-match-dist", type=float, default=1.0,
                   help="distância máx. entre os pés de um atleta em quadros analisados seguidos, "
                        "em alturas de caixa, para considerá-lo o mesmo (padrão 1)")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.frame_skip < 1:
        sys.exit("--frame-skip deve ser >= 1")
    run(a)
