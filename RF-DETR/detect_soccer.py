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
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import Counter
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

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
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
              "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y", "area", "team"]
POSSESSION_FIELDS = ["frame", "timestamp", "ball_detected", "team_near_ball", "possession_team",
                     "seconds_A", "seconds_B", "pct_A", "pct_B"]


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
    t0 = time.time()
    dt = args.frame_skip / fps  # segundos de vídeo representados por quadro analisado

    poss_fh = open(poss_path, "w", newline="") if poss_path else None
    poss_writer = csv.DictWriter(poss_fh, fieldnames=POSSESSION_FIELDS) if poss_fh else None
    if poss_writer:
        poss_writer.writeheader()

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
                for k, ((x1, y1, x2, y2), conf, cid) in enumerate(zip(det.xyxy, det.confidence, det.class_id)):
                    name = class_name(names, int(cid))
                    counts[name] += 1
                    w, h = x2 - x1, y2 - y1
                    writer.writerow({
                        "frame": idx, "timestamp": round(ts, 3), "class_id": int(cid),
                        "class_name": name, "confidence": round(float(conf), 4),
                        "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                        "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                        "width": round(float(w), 1), "height": round(float(h), 1),
                        "center_x": round(float(x1 + w / 2), 1), "center_y": round(float(y1 + h / 2), 1),
                        "area": round(float(w * h), 1),
                        "team": TEAM_SHORT[last_teams[k]] if last_teams[k] >= 0 else "",
                    })
            if sink and (run_model or args.reuse_detections):
                if tracker:
                    img = annotate(frame, last, names, not args.hide_conf, last_teams,
                                   teams_clf.colors, tracker.holder)
                    draw_possession_panel(img, tracker, teams_clf.colors)
                else:
                    img = annotate(frame, last, names, not args.hide_conf)
                sink.write(img)
            if run_model and frames_done % 10 == 0:
                el = time.time() - t0
                n_left = (end - idx - 1) / args.frame_skip
                print(f"  quadro {idx}/{end - 1} | {frames_done / el:.2f} q/s | "
                      f"restam ~{n_left * el / frames_done:.0f}s", flush=True)

    cap.release()
    if sink:
        sink.close()
    if poss_fh:
        poss_fh.close()

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
    return {"frames": frames_done, "counts": dict(counts), "csv": str(csv_path),
            "video": str(video_path) if sink else None, "possession": possession}


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
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.frame_skip < 1:
        sys.exit("--frame-skip deve ser >= 1")
    run(a)
