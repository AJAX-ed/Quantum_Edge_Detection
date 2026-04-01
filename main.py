# =============================================================================
# main.py  -  Quantum Edge Detection  (QPIE + FRQI demo)
# =============================================================================
# Run this file directly:  python main.py
# For a one-click setup + run on any machine:  python run.py
# =============================================================================

import numpy
from matplotlib import pyplot as plt
from PIL import Image

from quantumimageencoding.FRQI import FRQI
from quantumimageencoding.QPIE import QPIE
from utils import showdiff


def banner(title):
    """Print a visible section separator."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo image: 4x4 binary pattern (no external file required)
# ---------------------------------------------------------------------------
arr = numpy.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0]
], dtype=float)

# To use a real image from disk, uncomment and adjust the path below:
# from quantumimageencoding.QPIE import QPIE as _tmp
# _enc = _tmp()
# arr = numpy.array(_enc.preProcessImage(Image.open('./assets/images/test4edges.png').convert('L')), dtype=float)

# ===========================================================================
# SECTION 1: QPIE  (Quantum Probability Image Encoding)
# ===========================================================================
banner("SECTION 1 : QPIE Encoding")

Encoder_QPIE = QPIE()
qc_qpie = Encoder_QPIE.encode(arr)

print("[QPIE] Circuit encoded successfully.")
print(f"[QPIE] Number of qubits : {qc_qpie.num_qubits}")
print(f"[QPIE] Circuit depth    : {qc_qpie.depth()}")

banner("SECTION 1b: QPIE Circuit Diagram")
qc_qpie.draw(output='mpl', fold=-1)
plt.title("QPIE Encoding Circuit")
plt.tight_layout()
plt.show()

banner("SECTION 1c: QPIE Edge Detection")
qpie_edge_img = None
try:
    qpie_edge_img, h_img, v_img = Encoder_QPIE.detectEdges()
    print("[QPIE] Edge detection complete.")
    print(f"[QPIE] Edge image shape : {qpie_edge_img.shape}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(h_img, cmap='gray')
    axes[1].set_title('H-Edges (QPIE)')
    axes[2].imshow(v_img, cmap='gray')
    axes[2].set_title('V-Edges (QPIE)')
    axes[3].imshow(qpie_edge_img, cmap='gray')
    axes[3].set_title('Combined Edges (QPIE)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('QPIE Quantum Edge Detection', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[QPIE] Edge detection failed: {e}")
    print("  -> Ensure qiskit-aer is installed: pip install qiskit-aer>=0.14.0")

# ===========================================================================
# SECTION 2: FRQI  (Flexible Representation of Quantum Images)
# ===========================================================================
banner("SECTION 2 : FRQI Encoding")

Encoder_FRQI = FRQI()
qc_frqi = Encoder_FRQI.encode(arr)

print("[FRQI] Circuit encoded successfully.")
print(f"[FRQI] Number of qubits : {qc_frqi.num_qubits}")
print(f"[FRQI] Circuit depth    : {qc_frqi.depth()}")

banner("SECTION 2b: FRQI Circuit Diagram")
qc_frqi.draw(output='mpl', fold=-1)
plt.title("FRQI Encoding Circuit")
plt.tight_layout()
plt.show()

banner("SECTION 2c: FRQI Edge Detection")
frqi_edge_img = None
try:
    frqi_edge_img, h_img_f, v_img_f = Encoder_FRQI.detectEdges()
    print("[FRQI] Edge detection complete.")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(h_img_f, cmap='gray')
    axes[1].set_title('H-Edges (FRQI)')
    axes[2].imshow(v_img_f, cmap='gray')
    axes[2].set_title('V-Edges (FRQI)')
    axes[3].imshow(frqi_edge_img, cmap='gray')
    axes[3].set_title('Combined Edges (FRQI)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('FRQI Quantum Edge Detection', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[FRQI] Edge detection failed: {e}")
    print("  -> Ensure qiskit-aer is installed: pip install qiskit-aer>=0.14.0")

# ===========================================================================
# SECTION 3: Side-by-side comparison via showdiff
# ===========================================================================
banner("SECTION 3 : Comparison  (Original  vs  QPIE edges  vs  FRQI edges)")
if qpie_edge_img is not None and frqi_edge_img is not None:
    showdiff(Encoder_QPIE, arr, qpie_edge_img, frqi_edge_img)
elif qpie_edge_img is not None:
    showdiff(Encoder_QPIE, arr, qpie_edge_img)
else:
    showdiff(Encoder_QPIE, arr)

print("\n[Done] All steps completed. Close any open plot windows to exit.")
