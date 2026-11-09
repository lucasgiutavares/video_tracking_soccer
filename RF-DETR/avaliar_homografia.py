#!/usr/bin/env python3
"""
Métricas de qualidade da homografia do campo, a partir das saídas do detect_soccer.py --pitch.

  * cobertura: % de quadros analisados com homografia
  * origem da homografia por quadro e motivos de rejeição da observação
  * instabilidade: quanto o ponto do campo sob um pixel "anda" entre quadros seguidos,
    já descontado o movimento real da câmera (fluxo óptico) — em uma homografia
    perfeita seria ~0 m
  * posições de atletas removidas / interpoladas pelo filtro de velocidade

Exemplo:
  python avaliar_homografia.py ~/cbf/teste_corte23.mp4 saida/teste_corte23_pitch.csv \
      --detections saida/teste_corte23_detections.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter

import cv2
import numpy as np

from detect_soccer import CameraMotion, project_h

H_KEYS = [f"h{r}{c}" for r in (1, 2, 3) for c in (1, 2, 3)]


def load_pitch(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["frame"] = int(r["frame"])
        r["H"] = np.array([float(r[k]) for k in H_KEYS]).reshape(3, 3) if r["homography"] == "1" else None
    return rows


def instability(video: str, rows: list[dict], boxes: dict[int, np.ndarray]) -> np.ndarray:
    """Distância média (m) entre quadros seguidos dos pontos do campo sob pixels de prova, compensando a câmera."""
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probes = np.array([[W / 2, 0.75 * H], [W / 2, 0.55 * H], [0.2 * W, 0.9 * H], [0.8 * W, 0.9 * H]])
    motion = CameraMotion()
    by_frame = {r["frame"]: r for r in rows}
    out, prev_H, prev_idx = [], None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, rows[0]["frame"])
    for idx in range(rows[0]["frame"], rows[-1]["frame"] + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in by_frame:
            continue
        M = motion(frame, boxes.get(idx, np.zeros((0, 4))))
        cur = by_frame[idx]["H"]
        if cur is not None and prev_H is not None and M is not None and prev_idx is not None:
            a, _ = project_h(prev_H, probes)
            b, _ = project_h(cur, project_h(M, probes)[0])
            out.append(np.linalg.norm(a - b, axis=1).mean())
        prev_H, prev_idx = cur, idx
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("video", help="vídeo original analisado")
    p.add_argument("pitch_csv", help="CSV *_pitch.csv")
    p.add_argument("--detections", help="CSV *_detections.csv (filtro de velocidade e máscara dos atletas)")
    a = p.parse_args()

    rows = load_pitch(a.pitch_csv)
    n = len(rows)
    print(f"Quadros analisados: {n}")
    print(f"Com homografia:     {sum(r['H'] is not None for r in rows) / n:6.1%}")
    if "source" in rows[0]:
        print("Origem:             " + ", ".join(f"{k}={v}" for k, v in Counter(r["source"] for r in rows).most_common()))
        rej = Counter(r["rejected"] for r in rows if r["rejected"])
        print("Rejeições:          " + (", ".join(f"{k}={v}" for k, v in rej.most_common()) or "nenhuma"))

    boxes: dict[int, list] = {}
    flags = Counter()
    if a.detections:
        for r in csv.DictReader(open(a.detections)):
            if r["class_name"] != "ball":
                boxes.setdefault(int(r["frame"]), []).append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
                if r.get("pitch_x") or r.get("pitch_flag"):
                    flags[r.get("pitch_flag") or "ok"] += 1
    inst = instability(a.video, rows, {k: np.array(v) for k, v in boxes.items()})
    if len(inst):
        print(f"Instabilidade (m):  p50 {np.median(inst):.2f}  p95 {np.percentile(inst, 95):.2f}  "
              f"máx {inst.max():.2f}  (> 2 m em {np.mean(inst > 2):.1%} dos pares)")
    if flags:
        tot = sum(flags.values())
        print("Posições de atletas: " + ", ".join(f"{k} {v} ({v / tot:.0%})" for k, v in flags.most_common()))


if __name__ == "__main__":
    main()
