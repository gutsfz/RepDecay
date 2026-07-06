"""
RepDecay — Etapa 1: extração de landmarks via MediaPipe Pose.

Roda o MediaPipe Pose Landmarker num vídeo de agachamento (câmera lateral) e
extrai, frame a frame, os landmarks relevantes em DUAS representações:

  1. Landmarks normalizados — coordenadas (x, y) em fração da imagem [0..1].
     Servem para overlay visual e debug em pixels; NÃO servem para medir
     velocidade em m/s, porque dependem do enquadramento da câmera.
  2. World landmarks — coordenadas 3D em METROS reais, com origem no ponto
     médio dos quadris (documentação oficial do Pose Landmarker, Google AI
     Edge). São a base do cálculo de velocidade concêntrica em m/s (etapa 4).

Saídas:
  - data/landmarks/<video>_landmarks.csv  — uma linha por frame (formato wide)
  - data/debug/<video>_anotado.mp4        — vídeo com esqueleto sobreposto
  - data/debug/<video>_frames/*.png       — amostra de frames anotados

Uso:
  python src/extracao_pose.py data/videos_brutos/meu_video.mp4
  python src/extracao_pose.py video.mp4 --modelo heavy --sem-video-debug

Notas sobre o que é técnica estabelecida vs. decisão de design do projeto:
  - [ESTABELECIDO] O MediaPipe Pose e seus world landmarks são ferramenta
    documentada e usada em pesquisa de biomecânica, com erro conhecido de
    ~30-50 mm na localização de quadril/joelho vs. sistemas com marcador.
  - [DESIGN DO PROJETO] Usar o quadril como proxy do trajeto da barra é a
    parte EXPLORATÓRIA do RepDecay (nenhum estudo encontrado faz isso) e
    será auto-validada na etapa 5 contra marcação manual quadro a quadro.
    Este script só extrai o dado bruto; nenhuma inferência acontece aqui.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Console do Windows costuma usar cp850/cp1252; força UTF-8 pra saída acentuada
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ_PROJETO / "modelos_mediapipe"
DIR_LANDMARKS = RAIZ_PROJETO / "data" / "landmarks"
DIR_DEBUG = RAIZ_PROJETO / "data" / "debug"

# URLs oficiais dos modelos (Google AI Edge — gratuitos).
# lite < full < heavy em acurácia e custo computacional.
URLS_MODELO = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# Índices oficiais dos 33 landmarks do MediaPipe Pose.
# Extraímos só os articulares relevantes pro agachamento lateral
# (ombro entra porque é candidato a proxy alternativo da barra:
# no agachamento livre a barra apoia nas costas, na altura dos ombros).
LANDMARKS_RELEVANTES = {
    "ombro_esq": 11,
    "ombro_dir": 12,
    "quadril_esq": 23,
    "quadril_dir": 24,
    "joelho_esq": 25,
    "joelho_dir": 26,
    "tornozelo_esq": 27,
    "tornozelo_dir": 28,
}

N_FRAMES_AMOSTRA = 6  # frames PNG salvos pra inspeção visual rápida

# Conexões do esqueleto BlazePose (33 landmarks) — mesmo conjunto do
# POSE_CONNECTIONS do MediaPipe. Replicado aqui porque a wheel do
# mediapipe pro Python 3.13 traz só a Tasks API, sem mp.solutions
# (e portanto sem drawing_utils); o overlay é desenhado com OpenCV puro.
CONEXOES_POSE = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def baixar_modelo(variante: str) -> Path:
    """Baixa o arquivo .task do modelo se ainda não existir localmente."""
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_MODELOS / f"pose_landmarker_{variante}.task"
    if destino.exists():
        return destino
    url = URLS_MODELO[variante]
    print(f"Baixando modelo '{variante}' de {url} ...")
    urllib.request.urlretrieve(url, destino)
    print(f"Modelo salvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def criar_landmarker(caminho_modelo: Path) -> vision.PoseLandmarker:
    """
    Cria o PoseLandmarker em modo VIDEO.

    [DESIGN DO PROJETO] Usamos a Tasks API (atual) em vez da solução legada
    mp.solutions.pose (deprecada desde 2023). O modo VIDEO exige timestamps
    monotônicos e habilita rastreamento entre frames, o que dá séries
    temporais mais estáveis do que detectar cada frame isoladamente (modo
    IMAGE) — estabilidade importa porque as etapas 2-4 derivam velocidade
    desse sinal.
    """
    opcoes = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(caminho_modelo)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,  # só o atleta; evita capturar pessoas ao fundo
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opcoes)


def desenhar_esqueleto(frame_bgr: np.ndarray, landmarks_normalizados) -> np.ndarray:
    """
    Sobrepõe o esqueleto detectado no frame (validação visual).

    Landmarks usados pelo pipeline (ombro, quadril, joelho, tornozelo) são
    destacados em vermelho e maiores; o resto do esqueleto fica em verde.
    """
    anotado = frame_bgr.copy()
    alt, larg = anotado.shape[:2]
    pontos = [
        (int(lm.x * larg), int(lm.y * alt)) for lm in landmarks_normalizados
    ]
    for a, b in CONEXOES_POSE:
        cv2.line(anotado, pontos[a], pontos[b], (0, 200, 0), 2)
    indices_relevantes = set(LANDMARKS_RELEVANTES.values())
    for idx, pt in enumerate(pontos):
        if idx in indices_relevantes:
            cv2.circle(anotado, pt, 6, (0, 0, 255), -1)
        else:
            cv2.circle(anotado, pt, 3, (0, 200, 0), -1)
    return anotado


def extrair_video(
    caminho_video: Path,
    variante_modelo: str = "full",
    gerar_video_debug: bool = True,
) -> pd.DataFrame:
    """Processa o vídeo inteiro e retorna um DataFrame com uma linha por frame."""
    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS)
    n_frames_meta = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise ValueError(
            f"FPS inválido ({fps}) nos metadados do vídeo — sem FPS confiável "
            "não dá pra converter frames em segundos nem calcular velocidade."
        )
    print(f"Vídeo: {caminho_video.name} — {largura}x{altura}, {fps:.2f} fps, ~{n_frames_meta} frames")

    landmarker = criar_landmarker(baixar_modelo(variante_modelo))

    escritor_video = None
    dir_frames_amostra = None
    if gerar_video_debug:
        DIR_DEBUG.mkdir(parents=True, exist_ok=True)
        caminho_saida = DIR_DEBUG / f"{caminho_video.stem}_anotado.mp4"
        escritor_video = cv2.VideoWriter(
            str(caminho_saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
        )
        dir_frames_amostra = DIR_DEBUG / f"{caminho_video.stem}_frames"
        dir_frames_amostra.mkdir(parents=True, exist_ok=True)
        # Frames espaçados uniformemente pra amostra PNG
        indices_amostra = set(
            np.linspace(0, max(n_frames_meta - 1, 0), N_FRAMES_AMOSTRA, dtype=int).tolist()
        )

    linhas: list[dict] = []
    idx_frame = 0
    while True:
        ok, frame_bgr = captura.read()
        if not ok:
            break

        # Timestamp derivado do índice do frame e do FPS dos metadados.
        # É a mesma base de tempo que será usada pra velocidade (etapa 4).
        timestamp_ms = int(round(idx_frame * 1000.0 / fps))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = landmarker.detect_for_video(imagem_mp, timestamp_ms)

        detectou = bool(resultado.pose_landmarks)
        linha: dict = {
            "frame": idx_frame,
            "timestamp_ms": timestamp_ms,
            "deteccao": detectou,
        }

        if detectou:
            norm = resultado.pose_landmarks[0]
            world = resultado.pose_world_landmarks[0]
            for nome, idx in LANDMARKS_RELEVANTES.items():
                linha[f"{nome}_x_norm"] = norm[idx].x
                linha[f"{nome}_y_norm"] = norm[idx].y
                linha[f"{nome}_z_norm"] = norm[idx].z
                linha[f"{nome}_x_world"] = world[idx].x
                linha[f"{nome}_y_world"] = world[idx].y
                linha[f"{nome}_z_world"] = world[idx].z
                # visibility: confiança de que o ponto está visível na imagem.
                # Guardamos pra poder filtrar frames ruins nas etapas seguintes.
                linha[f"{nome}_visibilidade"] = norm[idx].visibility
        # Frames sem detecção viram NaN nas colunas de coordenadas — a linha
        # é mantida pra preservar a base de tempo contínua do sinal.

        if escritor_video is not None:
            anotado = (
                desenhar_esqueleto(frame_bgr, resultado.pose_landmarks[0])
                if detectou
                else frame_bgr
            )
            escritor_video.write(anotado)
            if idx_frame in indices_amostra:
                cv2.imwrite(str(dir_frames_amostra / f"frame_{idx_frame:05d}.png"), anotado)

        linhas.append(linha)
        idx_frame += 1
        if idx_frame % 100 == 0:
            print(f"  ... {idx_frame} frames processados")

    captura.release()
    if escritor_video is not None:
        escritor_video.release()
        print(f"Vídeo anotado salvo em {DIR_DEBUG / (caminho_video.stem + '_anotado.mp4')}")
        print(f"Frames de amostra em {dir_frames_amostra}")
    landmarker.close()

    return pd.DataFrame(linhas)


def resumo_extracao(df: pd.DataFrame) -> None:
    """Imprime evidência de que a extração funcionou (ou alertas se não)."""
    total = len(df)
    detectados = int(df["deteccao"].sum())
    taxa = detectados / total * 100 if total else 0.0
    print("\n===== RESUMO DA EXTRAÇÃO =====")
    print(f"Frames processados: {total}")
    print(f"Frames com pose detectada: {detectados} ({taxa:.1f}%)")

    if detectados:
        print("\nVisibilidade média por landmark (0 a 1; baixo = ponto oculto/instável):")
        for nome in LANDMARKS_RELEVANTES:
            col = f"{nome}_visibilidade"
            if col in df.columns:
                print(f"  {nome:15s} {df[col].mean():.3f}")

        # Sanidade do world landmark: a origem é o ponto médio dos quadris,
        # então o y_world do quadril deve oscilar perto de 0 e a amplitude
        # vertical num agachamento deve ficar na ordem de décimos de metro.
        y_med = (df["quadril_esq_y_world"] + df["quadril_dir_y_world"]) / 2
        print("\nSanidade (y_world do ponto médio dos quadris, em metros):")
        print(f"  min={y_med.min():.3f}  max={y_med.max():.3f}  amplitude={y_med.max() - y_med.min():.3f}")

    if taxa < 90:
        print(
            "\n[ALERTA] Taxa de detecção abaixo de 90%. Confira iluminação, "
            "enquadramento (corpo inteiro visível de perfil) e oclusões. "
            "Os apps mal avaliados na literatura falhavam justamente em perder "
            "repetições — queremos taxa de detecção alta antes de seguir adiante."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RepDecay etapa 1 — extração de landmarks")
    parser.add_argument("video", type=Path, help="Caminho do vídeo de agachamento (câmera lateral)")
    parser.add_argument(
        "--modelo",
        choices=list(URLS_MODELO),
        default="full",
        help="Variante do Pose Landmarker (default: full; use heavy pra máxima acurácia — "
        "V1 processa vídeo gravado, então custo de tempo não é crítico)",
    )
    parser.add_argument(
        "--sem-video-debug",
        action="store_true",
        help="Não gerar vídeo anotado nem frames de amostra (mais rápido)",
    )
    args = parser.parse_args()

    df = extrair_video(args.video, args.modelo, gerar_video_debug=not args.sem_video_debug)

    DIR_LANDMARKS.mkdir(parents=True, exist_ok=True)
    caminho_csv = DIR_LANDMARKS / f"{args.video.stem}_landmarks.csv"
    df.to_csv(caminho_csv, index=False)
    print(f"\nCSV de landmarks salvo em {caminho_csv}")

    resumo_extracao(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
