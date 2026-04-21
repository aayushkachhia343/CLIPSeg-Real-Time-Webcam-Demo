# CLIPSeg Real-Time Webcam Demo

A minimal, interactive demo of [CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg) — zero-shot text-prompted image segmentation running live on your webcam.

Type any word. Watch the model find it.

---

## What is CLIPSeg?

CLIPSeg is a vision-language model that segments *any* object you can describe in plain English — no retraining, no labeled data. You give it a text prompt like `"cup"` or `"person's hand"`, and it produces a pixel-level probability map showing where that thing is in the image.

It works by combining two ideas:
- **CLIP** (Contrastive Language–Image Pretraining): maps images and text into the same vector space so that similar concepts cluster together
- **A lightweight decoder** on top of CLIP's image features that upsamples patch-level embeddings into a dense segmentation mask

---

## The Math

**Step 1 — CLIP embeds image and text into a shared space:**

```
image_features = CLIP_encoder(image)       # shape: (N_patches, D)
text_features  = CLIP_encoder(text_prompt) # shape: (D,)
```

CLIP is trained with contrastive loss so that `cos(image_features, text_features)` is high when the image matches the text.

**Step 2 — CLIPSeg's decoder produces a dense prediction:**

For each pixel location `(i, j)`, the decoder attends over image patch features conditioned on the text embedding:

```
logit(i, j) = Decoder(image_features, text_features)[i, j]
```

**Step 3 — Sigmoid converts logits to probabilities:**

```
p(i, j) = σ(logit(i, j)) = 1 / (1 + exp(-logit(i, j)))
```

`p(i, j) ∈ [0, 1]` is how confident the model is that pixel `(i, j)` belongs to the prompted concept.

**What this demo visualizes:**

- **Heatmap mode**: `p(i,j)` mapped to a color — blue = low confidence, red = high
- **Contours mode**: binary mask at `p(i,j) ≥ threshold`, outline drawn
- **Both**: heatmap + contours overlaid

---

## Key Points

| Property | Detail |
|----------|--------|
| Zero-shot | No fine-tuning needed — prompts drive the prediction |
| Open-vocabulary | Anything you can describe in English |
| Backbone | CLIP ViT-B/16 (vision transformer) |
| Output | 352×352 logit map, upsampled to frame resolution |
| Weakness | Coarse boundaries — trained on weak supervision, not pixel-perfect masks |
| Speed | ~0.5–2s/frame depending on GPU/CPU |

**Prompt tips:**
- Specific concrete nouns work best: `"coffee cup"` > `"object"`
- Body parts: `"left hand"`, `"face"`, `"eye"`
- Colors help: `"red shirt"` > `"shirt"`
- Avoid abstract concepts: `"happiness"` won't localize to anything useful

---

## When to Use CLIPSeg

**Good fit:**
- You don't know the object categories ahead of time
- You want to prototype a segmentation idea in minutes
- You want text-controlled visual attention for demos or research
- You need to roughly locate objects without labeled training data

**Not a good fit:**
- You need pixel-perfect masks → use SAM or Mask R-CNN
- You need real-time speed (>30 FPS) → use a dedicated single-class model
- You have labeled data for a fixed set of classes → fine-tune a standard segmentation model

---

## Alternative Models

| Model | Strengths | Weakness vs CLIPSeg |
|-------|-----------|---------------------|
| [SAM (Meta)](https://github.com/facebookresearch/segment-anything) | Near-perfect masks, fast with GPU | Needs point/box prompt, not text |
| [SAM 2](https://github.com/facebookresearch/sam2) | SAM + video tracking | Same — no text prompt natively |
| [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Text → box (DINO) + mask (SAM) = best quality | Two models, heavier setup |
| [OWL-ViT / OWLv2](https://huggingface.co/google/owlvit-base-patch32) | Open-vocab detection (boxes), very fast | Detection only, no masks |
| [FC-CLIP](https://github.com/bytedance/fc-clip) | Better accuracy than CLIPSeg, faster | Harder to set up |
| [ODISE](https://github.com/NVlabs/ODISE) | Diffusion-based, high-quality masks | Very heavy, slow |

**Practical recommendation:** For production text-prompted segmentation, use **Grounded SAM**. For rapid experimentation and demos, use **CLIPSeg** (this repo).

---

## Use Cases

- **Dataset annotation**: use text to pre-label regions, then refine manually
- **Robotics**: segment a target object by name without retraining the robot's vision system
- **AR/VR**: overlay effects on objects described in text, driven by voice commands
- **Content moderation**: detect specific visual concepts in images
- **Medical imaging**: highlight regions of interest described in clinical language

---

## Install

```bash
git clone https://github.com/YOUR_USERNAME/clipseg-webcam-demo
cd clipseg-webcam-demo
pip install -r requirements.txt
```

GPU is recommended but not required. On CPU, inference takes ~1–2 seconds per frame (the video stays live between frames thanks to the async capture thread).

---

## Run

```bash
python app.py
```

The model (~330 MB) downloads automatically on first run via Hugging Face Hub.

---

## Controls

| Key | Action |
|-----|--------|
| `e` | Edit prompts (type comma-separated list in terminal, press Enter) |
| `c` | Cycle visualization mode: heatmap → contours → both |
| `+` / `=` | Increase detection threshold (+0.05) |
| `-` | Decrease detection threshold (−0.05) |
| `]` | Increase overlay blend ratio (+0.05) |
| `[` | Decrease overlay blend ratio (−0.05) |
| `s` | Save snapshot PNG to current directory |
| `q` | Quit |

---

## How It Works (Code Walkthrough)

```
app.py
├── PROMPT_COLORS          — per-prompt BGR colors (red, green, blue)
├── build_overlay()        — builds heatmap/contour/both visualization
├── render_hud()           — draws on-screen text (prompts, FPS, mode)
├── capture_loop()         — daemon thread: reads webcam → shared slot
├── run_inference()        — runs CLIPSeg, returns (N, H, W) prob array
└── main()
    ├── Load model
    ├── Start capture thread
    └── Loop: read frame → infer → overlay → show → handle keys
```

The **slot pattern** (not a queue) is the key performance trick: the capture thread always overwrites the same variable with the freshest frame. The inference thread always works on the latest image, not a backlog of stale frames.

---

## License

MIT
