import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Adiciona o diretório raiz ao PATH (para importar src mesmo fora da pasta)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classical.thresholding import otsu_segment, adaptive_segment
from src.classical.kmeans_segmentation import kmeans_segment
from src.classical.watershed import watershed_segment
from src.deep.unet_inference import unet_infer
from src.deep.maskrcnn_inference import maskrcnn_infer
from src.utils.visualization import overlay_mask, show_side_by_side
from src.utils.metrics import compute_iou, compute_dice

st.set_page_config("Image Segmentation Playground", layout="wide")
st.title("🧠 Image Segmentation Playground — Streamlit")

uploaded = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg"])
method = st.selectbox("Método", [
    "Otsu Threshold",
    "Adaptive Threshold",
    "Watershed",
    "K-Means",
    "U-Net (PyTorch, CPU)",
    "Mask R-CNN (torchvision)"
])

with st.expander("⚙️ Parâmetros"):
    if method == "Adaptive Threshold":
        block_size = st.slider("blockSize (ímpar)", 3, 101, 35, step=2)
        C = st.slider("C", 0, 20, 5)
    if method == "K-Means":
        k = st.slider("k (clusters)", 2, 10, 3)
    if method == "U-Net (PyTorch, CPU)":
        unet_checkpoint = st.text_input("Caminho para checkpoint (opcional)", "")
        thresh_unet = st.slider("Binarizar probabilidade >", 0.1, 0.9, 0.5)
    if method == "Mask R-CNN (torchvision)":
        score_thresh = st.slider("Score threshold", 0.0, 1.0, 0.5)

if uploaded is None:
    st.info("📁 Envie uma imagem para começar.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

st.markdown("### 🖼️ Imagem Original")
st.image(img_rgb, caption="Original", width="stretch")

mask = None
if method == "Otsu Threshold":
    mask = otsu_segment(img_gray)
elif method == "Adaptive Threshold":
    mask = adaptive_segment(img_gray, block_size=block_size, C=C)
elif method == "Watershed":
    mask = watershed_segment(img_bgr)
elif method == "K-Means":
    mask = kmeans_segment(img_rgb, k=k)
elif method == "U-Net (PyTorch, CPU)":
    mask_prob = unet_infer(img_rgb, checkpoint_path=unet_checkpoint)
    mask = (mask_prob > thresh_unet).astype("uint8")
elif method == "Mask R-CNN (torchvision)":
    masks, scores = maskrcnn_infer(img_rgb, score_threshold=score_thresh)
    if len(masks) > 0:
        mask = np.any(np.stack(masks, axis=0), axis=0).astype("uint8")
    else:
        mask = np.zeros(img_gray.shape, dtype="uint8")

st.markdown("### 🎯 Resultado da Segmentação")
if mask is not None:
    if mask.ndim == 3 and mask.shape[2] == 3:
        st.image(mask, caption="Máscara (RGB)", width="stretch")
    else:
        mask_bin = (mask > 0).astype("uint8")
        overlay = overlay_mask(img_rgb, mask_bin, color=(255, 0, 0), alpha=0.4)
        show_side_by_side(img_rgb, overlay, "Original", "Overlay")
        st.image(mask_bin * 255, caption="Máscara Binária", width="stretch")

gt_file = st.file_uploader(
    "📊 Upload ground-truth mask (opcional, grayscale 0/255)",
    type=["png", "jpg", "jpeg"],
    key="gt"
)

if gt_file is not None and mask is not None:
    gt_bytes = np.asarray(bytearray(gt_file.read()), dtype=np.uint8)
    gt_bgr = cv2.imdecode(gt_bytes, cv2.IMREAD_COLOR)
    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
    gt_bin = (gt_gray > 127).astype("uint8")

    iou = compute_iou(gt_bin, (mask > 0).astype("uint8"))
    dice = compute_dice(gt_bin, (mask > 0).astype("uint8"))
    st.success(f"IoU: **{iou:.4f}** — Dice: **{dice:.4f}**")

st.write("---")
st.caption("U-Net em PyTorch (CPU) e Mask R-CNN do torchvision.")
