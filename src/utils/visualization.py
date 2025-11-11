import numpy as np
import cv2
import streamlit as st

def overlay_mask(img_rgb, mask, color=(255,0,0), alpha=0.4):
    # img_rgb uint8 HxWx3, mask 0/1 HxW
    overlay = img_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask==1] = overlay[mask==1] * (1-alpha) + color_arr * alpha
    return overlay.astype(np.uint8)

def show_side_by_side(img1, img2, lbl1="A", lbl2="B"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].imshow(img1); axes[0].set_title(lbl1); axes[0].axis('off')
    axes[1].imshow(img2); axes[1].set_title(lbl2); axes[1].axis('off')
    st.pyplot(fig)
