from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk
import torch

from worldmodel.train import TrainConfig, train_worldsplat
from worldmodel.vkitti2 import VKITTI2_DEFAULT_VARIATIONS, discover_vkitti2
from worldmodel.worldsplat import WorldSplatVAE, depth_to_unit


class WorldStudio:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("WorldModel — WorldSplat VKITTI2 — RAY FIX")
        root.geometry("1180x820")
        self.messages: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.train_thread = None
        self.model: WorldSplatVAE | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z0 = None
        self.z1 = None
        self.photo = None
        self.last_input = None
        self._build_ui()
        self.root.after(100, self._poll)

    def _build_ui(self):
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(outer, padding=10)
        view = ttk.Frame(outer, padding=10)
        outer.add(controls, weight=0)
        outer.add(view, weight=1)

        ttk.Label(controls, text="WorldSplat — VKITTI2 RAY FIX", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(
            controls,
            text=(
                "Phase-1 A/B: image-plane ray position is now independent of depth. "
                "Same VKITTI2 RGB + metric-depth experiment; no stereo/world binding yet."
            ),
            wraplength=330,
        ).pack(anchor="w", pady=(0, 8))

        self.data_var = tk.StringVar()
        self.depth_var = tk.StringVar()
        self.vkitti_var = tk.StringVar()
        self.vkitti_info = tk.StringVar(value="Virtual KITTI 2 mode: OFF")
        self.out_var = tk.StringVar(value=str(Path.cwd() / "runs" / "vkitti2_rayfix"))
        # Strict A/B defaults copied from the completed 80k control run.
        self.steps_var = tk.IntVar(value=80000)
        self.size_var = tk.IntVar(value=128)
        self.splats_var = tk.IntVar(value=512)
        self.latent_var = tk.IntVar(value=96)
        self.batch_var = tk.IntVar(value=6)

        train_box = ttk.LabelFrame(controls, text="Train", padding=8)
        train_box.pack(fill=tk.X, pady=4)

        vkrow = ttk.Frame(train_box)
        vkrow.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(vkrow, text="LOAD VIRTUAL KITTI 2", command=self.load_vkitti2).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(vkrow, text="CLEAR", width=7, command=self.clear_vkitti2).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(train_box, textvariable=self.vkitti_info, wraplength=310).pack(anchor="w", pady=(0, 6))

        self._path_row(train_box, "Images", self.data_var, clear_vkitti=True)
        self._path_row(train_box, "Depth (optional)", self.depth_var, clear_vkitti=True)
        self._path_row(train_box, "Output", self.out_var)
        for label, var in [
            ("Steps", self.steps_var),
            ("Image size", self.size_var),
            ("Splats", self.splats_var),
            ("Latent", self.latent_var),
            ("Batch", self.batch_var),
        ]:
            row = ttk.Frame(train_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)
        row = ttk.Frame(train_box)
        row.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(row, text="START TRAIN", command=self.start_train).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="STOP", command=self.stop_train).pack(side=tk.LEFT, padx=(4, 0))
        self.progress = ttk.Progressbar(train_box, maximum=100)
        self.progress.pack(fill=tk.X, pady=(6, 2))
        self.status = tk.StringVar(value=f"ray-fix branch | device: {self.device}")
        ttk.Label(train_box, textvariable=self.status, wraplength=310).pack(anchor="w")

        explore = ttk.LabelFrame(controls, text="Explore", padding=8)
        explore.pack(fill=tk.X, pady=8)
        ttk.Button(explore, text="LOAD CHECKPOINT", command=self.load_checkpoint).pack(fill=tk.X, pady=2)
        ttk.Button(explore, text="NEW RANDOM WORLD", command=self.new_random).pack(fill=tk.X, pady=2)
        ttk.Button(explore, text="ENCODE IMAGE", command=self.encode_image).pack(fill=tk.X, pady=2)

        self.morph = tk.DoubleVar(value=0.0)
        self.yaw = tk.DoubleVar(value=0.0)
        self.pitch = tk.DoubleVar(value=0.0)
        self.focal = tk.DoubleVar(value=1.25)
        for label, var, lo, hi in [
            ("Morph", self.morph, 0, 1),
            ("Yaw", self.yaw, -30, 30),
            ("Pitch", self.pitch, -20, 20),
            ("Focal", self.focal, 0.8, 1.8),
        ]:
            ttk.Label(explore, text=label).pack(anchor="w")
            ttk.Scale(explore, variable=var, from_=lo, to=hi, command=lambda _=None: self.render()).pack(fill=tk.X)
        ttk.Button(explore, text="RESET CAMERA", command=self.reset_camera).pack(fill=tk.X, pady=(6, 2))
        ttk.Label(
            explore,
            text=(
                "Left: belief render   |   Right: learned depth\n"
                "Judge geometry first with preview_latest.png or ENCODE IMAGE. "
                "NEW RANDOM WORLD is a separate prior-support test.\n"
                "VKITTI2 depth is fixed-scale (cm -> metres -> /80 m)."
            ),
            wraplength=310,
        ).pack(anchor="w", pady=(6, 0))

        self.canvas_label = ttk.Label(view, anchor="center")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)
        self.info = tk.StringVar(value="Load Virtual KITTI 2, a normal folder, or a ray-fix checkpoint.")
        ttk.Label(view, textvariable=self.info, anchor="center").pack(fill=tk.X)

    def _path_row(self, parent, label, var, *, clear_vkitti=False):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def browse():
            p = filedialog.askdirectory()
            if p:
                if clear_vkitti:
                    self.clear_vkitti2(silent=True)
                var.set(p)

        ttk.Button(row, text="…", width=3, command=browse).pack(side=tk.LEFT, padx=(3, 0))

    def load_vkitti2(self):
        p = filedialog.askdirectory(
            title="Choose common parent containing extracted Virtual KITTI 2 RGB and depth archives"
        )
        if not p:
            return
        self.status.set("scanning Virtual KITTI 2…")
        self.root.update_idletasks()
        try:
            samples = discover_vkitti2(p, camera=0)
        except Exception as e:
            self.status.set("Virtual KITTI 2 load failed")
            messagebox.showerror(
                "Virtual KITTI 2",
                f"{e}\n\nChoose the COMMON PARENT containing both extracted RGB and depth trees.",
            )
            return

        self.vkitti_var.set(p)
        self.data_var.set("")
        self.depth_var.set("")
        self.out_var.set(str(Path.cwd() / "runs" / "vkitti2_rayfix"))
        # Strict A/B against the completed control run.
        self.steps_var.set(80000)
        self.size_var.set(128)
        self.splats_var.set(512)
        self.latent_var.set(96)
        self.batch_var.set(6)
        variations = ", ".join(VKITTI2_DEFAULT_VARIATIONS)
        self.vkitti_info.set(f"VKITTI2 ON: {len(samples)} paired frames | Camera_0 | {variations}")
        self.status.set(
            f"RAY FIX ready: {len(samples)} RGB+depth pairs; fixed depth scale 80 m; output kept separate"
        )

    def clear_vkitti2(self, *, silent=False):
        self.vkitti_var.set("")
        self.vkitti_info.set("Virtual KITTI 2 mode: OFF")
        if not silent:
            self.status.set(f"custom folder mode | ray fix | device: {self.device}")

    def start_train(self):
        if self.train_thread and self.train_thread.is_alive():
            return
        vkitti = self.vkitti_var.get().strip()
        if not vkitti and not self.data_var.get():
            messagebox.showerror("WorldSplat", "Load Virtual KITTI 2 or choose an image folder first.")
            return
        self.stop_event.clear()
        cfg = TrainConfig(
            data_dir=self.data_var.get(),
            depth_dir=self.depth_var.get() or None,
            vkitti2_root=vkitti or None,
            vkitti2_depth_max_m=80.0,
            vkitti2_camera=0,
            out_dir=self.out_var.get(),
            steps=int(self.steps_var.get()),
            image_size=int(self.size_var.get()),
            num_splats=int(self.splats_var.get()),
            latent_dim=int(self.latent_var.get()),
            batch=int(self.batch_var.get()),
        )

        def worker():
            try:
                train_worldsplat(cfg, callback=self.messages.put, stop_event=self.stop_event)
            except Exception as e:
                self.messages.put({"kind": "error", "error": repr(e)})

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()
        self.status.set("training ray-fix A/B…")

    def stop_train(self):
        self.stop_event.set()
        self.status.set("stop requested")

    def _poll(self):
        try:
            while True:
                m = self.messages.get_nowait()
                k = m.get("kind")
                if k == "cache":
                    self.status.set(f"caching {m['done']}/{m['total']}")
                elif k == "step":
                    self.progress["value"] = 100.0 * m["step"] / max(m["steps"], 1)
                    d = f" depth={m['depth']:.4f}" if m.get("has_depth") else " depth=UNSUPERVISED"
                    kind = m.get("dataset_kind", "folder")
                    self.status.set(f"RAY FIX | {kind} | step {m['step']}/{m['steps']} loss={m['loss']:.4f} rgb={m['rgb']:.4f}{d}")
                elif k == "preview":
                    self._show_image(Path(m["path"]))
                elif k == "done":
                    self.status.set(f"done: {m['path']}")
                    self._load_model(Path(m["path"]))
                elif k == "error":
                    self.status.set(m["error"])
                    messagebox.showerror("WorldSplat training", m["error"])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _show_image(self, path: Path):
        try:
            im = Image.open(path).convert("RGB")
            self._set_canvas(im)
        except Exception:
            pass

    def _set_canvas(self, im: Image.Image):
        w = max(self.canvas_label.winfo_width(), 400)
        h = max(self.canvas_label.winfo_height(), 400)
        im.thumbnail((w, h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        self.canvas_label.configure(image=self.photo)

    def load_checkpoint(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")])
        if p:
            self._load_model(Path(p))

    def _load_model(self, path: Path):
        try:
            self.model, extra = WorldSplatVAE.load_checkpoint(path, device=self.device)
            self.new_random()
            ds = extra.get("dataset_size", "?")
            dep = extra.get("depth_supervised", False)
            kind = extra.get("dataset_kind", "unknown")
            self.info.set(
                f"RAY FIX | {path.name} | dataset={ds} | kind={kind} | depth supervised={dep} | device={self.device}"
            )
        except Exception as e:
            messagebox.showerror("WorldSplat", repr(e))

    def new_random(self):
        if self.model is None:
            return
        d = self.model.cfg.latent_dim
        self.z0 = torch.randn(1, d, device=self.device)
        self.z1 = torch.randn(1, d, device=self.device)
        self.morph.set(0.0)
        self.render()

    def encode_image(self):
        if self.model is None:
            return
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*")])
        if not p:
            return
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (self.model.cfg.image_size, self.model.cfg.image_size), method=Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8).copy()
        x = torch.from_numpy(arr).to(self.device, torch.float32).permute(2, 0, 1)[None] / 255.0
        with torch.no_grad():
            mu, _ = self.model.encode(x)
        self.z0 = mu
        self.z1 = mu.clone()
        self.last_input = im.copy()
        self.morph.set(0.0)
        self.render()

    def reset_camera(self):
        self.yaw.set(0.0)
        self.pitch.set(0.0)
        self.focal.set(1.25)
        self.render()

    def render(self):
        if self.model is None or self.z0 is None:
            return
        a = float(self.morph.get())
        z = (1 - a) * self.z0 + a * self.z1
        with torch.no_grad():
            out = self.model.render_latent(
                z,
                yaw_deg=float(self.yaw.get()),
                pitch_deg=float(self.pitch.get()),
                focal=float(self.focal.get()),
            )
            rgb = (out.rgb[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            d = depth_to_unit(out.depth, self.model.cfg)[0].cpu().numpy()
        dg = ((1.0 - d) * 255).astype(np.uint8)
        depth_rgb = np.stack([dg, dg, dg], axis=-1)
        combo = np.concatenate([rgb, depth_rgb], axis=1)
        self._set_canvas(
            Image.fromarray(combo).resize((combo.shape[1] * 4, combo.shape[0] * 4), Image.Resampling.NEAREST)
        )


def main():
    root = tk.Tk()
    WorldStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
