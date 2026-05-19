## Semantic video editing with Attention-locked diffusion

This project implements a pipeline for text-guided appearance editing of objects in a video while preserving visual consistency across frames.

Built on top of Grounded SAM 2, RAFT and Stable Diffusion inpainting.

### Motivation
Native per-frame editing may suffer from flickering, drifting and the model re-deciding what the object boundaries are on every frame independently.

The pipeline tries to enforce three consistency mechanisms:
1. **Mask propagation via SAM 2**, allowing the editor to track the object across frames without redetection.
2. **Latent warping via RAFT optical flow**, used to initialize the diffusion on the frame from the previous frame's results, warped by motion.
3. **Cross-frame attention locking**, propagating the fixed diffusion model's internal attention map across frames.

### Example
| | |
|---|---|
| **Sequence** | DAVIS 2017 `blackswan` |
| **Edit prompt** | `"a white swan with golden wings"` |
| **Object** | black swan gliding across a dark lake |

<img width="1354" height="284" alt="Screenshot_20260519_190520" src="https://github.com/user-attachments/assets/1822dbee-6383-45a4-9e4e-7b568e3e9ff0" />

The swan is located with GroundingDINO, SAM 2 is then used to obtain the binary mask.

<img width="1356" height="394" alt="Screenshot_20260519_190743" src="https://github.com/user-attachments/assets/84cf6c0d-6e9f-4f39-b7b8-08f80a75afdb" />

The mask is propagated using SAM 2 predictor.

<img width="1366" height="448" alt="Screenshot_20260519_191012" src="https://github.com/user-attachments/assets/dd053a09-2531-4ecf-8bcf-2cb6d4e0955b" />

Intersection over Union (IoU) is consistently high, however, it starts to degrade toward the last frames, possibly error accumulation.

*coming soon...*

### Architecture
```
Input video
    ↓
Frame extraction (OpenCV)
    ↓
Object grounding (GroundingDINO)
    ↓
Anchor frame segmentation (SAM 2 image predictor)
    ↓
Mask propagation across all frames (SAM 2 video predictor)
    ↓
Optical flow computation (RAFT)                                      in progress
    ↓
Latent warping + diffusion editing (Stable Diffusion inpaint)        in progress
    ↓
Cross-frame attention locking (custom UNet hook)                     todo
    ↓
Video reconstruction (ffmpeg)                                        todo
```
### Dataset
**DAVIS 2017 at 480p resolution.**

It comes with frame segmentation masks, which are used to evaluate IoU of the propagated mask against the ground truth.

### Results (so far)
SAM 2 model propagated the anchor mask across all frames with mean IoU 0.95, indicating near-perfect mask stability.
### Evaluation
1. Learned Perceptual Image Patch Similarity (LPIPS) will measure flickering of frame-wise diffusion versus mask propagation only versus full pipeline.
2. IoU will measure mask stability.
3. Identity drift will be evaluated using the variance of CLIP embeddings of the object crop across frames.
### Ablation
* No mask propagation
* No latent warping
* No attention locking
* Full pipeline

### TODO
- [x]  extract frames (see `notebooks/00_frame_extraction.ipynb`)
- [x]  ground a frame with GroundingDINO and segment (see `notebooks/01_grounding_and_segmentation.ipynb`)
- [x]  propagate the mask (see `notebooks/02_mask_propagation.ipynb`)
- [ ]  calculate RAFT optical flow
- [ ]  SD inpainting with warm start latent initialization
- [ ]  implement custom diffusers with cross-frame locking
- [ ]  evaluation
- [ ]  visuals
