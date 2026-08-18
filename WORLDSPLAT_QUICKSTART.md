# WorldSplat quick start

```bash
pip install -r requirements-worldsplat.txt
python world_studio.py
```

For meaningful shallow 3-D rather than an RGB-only cardboard baseline, precompute relative depth once:

```bash
pip install transformers accelerate
python tools/make_depths.py --data D:/world_images --out D:/world_depth
python world_studio.py
```

In the GUI choose `D:/world_images`, optionally `D:/world_depth`, choose an output folder, then **START TRAIN**.

Start small on a 12 GB GPU:

```text
64 px
128 splats
latent 64
batch 8–12
10k–20k steps
```

The model is a scene-prior learner, not yet a persistent SLAM world. Single-view + monocular depth is explicitly **2.5-D**. The next gate is true multiview consistency.
