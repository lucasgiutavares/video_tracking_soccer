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


def annotate(frame: np.ndarray, det: Detections, names: list[str], show_conf: bool = True) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    thick = max(2, h // 540)
    font_scale = max(0.5, h / 1600)
    for (x1, y1, x2, y2), conf, cid in zip(det.xyxy, det.confidence, det.class_id):
        name = class_name(names, int(cid))
        color = CLASS_COLORS.get(name, FALLBACK_COLOR)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        if name == "ball":
            # triângulo acima da bola — mais visível que uma caixa minúscula
            cx, s = (x1 + x2) // 2, max(10, h // 90)
            pts = np.array([[cx, y1 - 4], [cx - s, y1 - 4 - 2 * s], [cx + s, y1 - 4 - 2 * s]])
            cv2.fillPoly(out, [pts], color)
            cv2.polylines(out, [pts], True, (0, 0, 0), max(1, thick // 2))
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = LABELS_PT.get(name, name) + (f" {conf:.2f}" if show_conf else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        txt_color = (255, 255, 255) if name in ("referee", "player") else (0, 0, 0)
        cv2.putText(out, label, (x1 + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    txt_color, thick, cv2.LINE_AA)
    return out


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
              "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y", "area"]


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
    t0 = time.time()

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
                det = (detect_tiled(detector, frame, base_thr, args.tiles, args.tile_overlap)
                       if args.tiles > 1 else detector(frame, base_thr))
                infer_time += time.time() - ti
                det = det.filter(det.confidence >= args.threshold) if not per_class else \
                    apply_class_thresholds(det, names, {n: per_class.get(n, args.threshold) for n in names})
                if args.single_ball:
                    det = keep_best_ball(det, names)
                last = det
                frames_done += 1
                ts = idx / fps
                for (x1, y1, x2, y2), conf, cid in zip(det.xyxy, det.confidence, det.class_id):
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
                    })
            if sink and (run_model or args.reuse_detections):
                sink.write(annotate(frame, last, names, not args.hide_conf))
            if run_model and frames_done % 10 == 0:
                el = time.time() - t0
                n_left = (end - idx - 1) / args.frame_skip
                print(f"  quadro {idx}/{end - 1} | {frames_done / el:.2f} q/s | "
                      f"restam ~{n_left * el / frames_done:.0f}s", flush=True)

    cap.release()
    if sink:
        sink.close()

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
    return {"frames": frames_done, "counts": dict(counts), "csv": str(csv_path),
            "video": str(video_path) if sink else None}


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
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.frame_skip < 1:
        sys.exit("--frame-skip deve ser >= 1")
    run(a)
