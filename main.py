# =============================================================================
# main.py  -  Quantum Edge Detection  (QPIE + FRQI demo)
# =============================================================================
# Run this file directly:  python main.py
# For a one-click setup+run use:  python run.py
# =============================================================================

import sys
import numpy
from matplotlib import pyplot as plt
from PIL import Image

from quantumimageencoding.FRQI import FRQI
from quantumimageencoding.QPIE import QPIE
from utils import showdiff

# ---------------------------------------------------------------------------
# Helper: print a section banner
# ---------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# Demo image  (4x4 binary pattern  - works on any machine, no file needed)
# ---------------------------------------------------------------------------
arr = numpy.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0]
], dtype=float)

# Uncomment below to use a real image from disk instead:
# raw = Image.open('./assets/images/test4edges.png').convert('L')
# Encoder_tmp = QPIE()
# arr_img = Encoder_tmp.preProcessImage(raw)
# arr = numpy.array(arr_img, dtype=float)

# ===========================================================================
# SECTION 1: QPIE  (Quantum Probability Image Encoding)
# ===========================================================================
banner("SECTION 1 : QPIE Encoding")

Encoder_QPIE = QPIE()
qc_qpie = Encoder_QPIE.encode(arr)

print("[QPIE] Circuit encoded successfully.")
print(f"[QPIE] Number of qubits : {qc_qpie.num_qubits}")
print(f"[QPIE] Circuit depth    : {qc_qpie.depth()}")

# Draw the QPIE circuit
banner("SECTION 1b: QPIE Circuit Diagram")
qc_qpie.draw(output='mpl', fold=-1)
plt.title("QPIE Encoding Circuit")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# QPIE Edge Detection
# ---------------------------------------------------------------------------
banner("SECTION 1c: QPIE Edge Detection")
try:
    edge_img, h_img, v_img = Encoder_QPIE.detectEdges()
    print("[QPIE] Edge detection complete.")
    print(f"[QPIE] Edge image shape : {edge_img.shape}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(h_img, cmap='gray')
    axes[1].set_title('Horizontal Edges (QPIE)')
    axes[2].imshow(v_img, cmap='gray')
    axes[2].set_title('Vertical Edges (QPIE)')
    axes[3].imshow(edge_img, cmap='gray')
    axes[3].set_title('Combined Edges (QPIE)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('QPIE Quantum Edge Detection', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[QPIE] Edge detection failed: {e}")
    print("  -> Make sure qiskit-aer is installed: pip install qiskit-aer>=0.14.0")

# ===========================================================================
# SECTION 2: FRQI  (Flexible Representation of Quantum Images)
# ===========================================================================
banner("SECTION 2 : FRQI Encoding")

Encoder_FRQI = FRQI()
qc_frqi = Encoder_FRQI.encode(arr)

print("[FRQI] Circuit encoded successfully.")
print(f"[FRQI] Number of qubits  : {qc_frqi.num_qubits}")
print(f"[FRQI] Circuit depth     : {qc_frqi.depth()}")

# Draw the FRQI circuit
banner("SECTION 2b: FRQI Circuit Diagram")
qc_frqi.draw(output='mpl', fold=-1)
plt.title("FRQI Encoding Circuit")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# FRQI Edge Detection
# ---------------------------------------------------------------------------
banner("SECTION 2c: FRQI Edge Detection")
try:
    edge_img_f, h_img_f, v_img_f = Encoder_FRQI.detectEdges()
    print("[FRQI] Edge detection complete.")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(h_img_f, cmap='gray')
    axes[1].set_title('Horizontal Edges (FRQI)')
    axes[2].imshow(v_img_f, cmap='gray')
    axes[2].set_title('Vertical Edges (FRQI)')
    axes[3].imshow(edge_img_f, cmap='gray')
    axes[3].set_title('Combined Edges (FRQI)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('FRQI Quantum Edge Detection', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[FRQI] Edge detection failed: {e}")
    print("  -> Make sure qiskit-aer is installed: pip install qiskit-aer>=0.14.0")

# ===========================================================================
# SECTION 3: Side-by-side comparison  (original vs edges)
# ===========================================================================
banner("SECTION 3 : Summary")
print("All encoding and edge-detection steps completed.")
print("Close any open plot windows to exit.")
showdiff(Encoder_QPIE, arr)
