# WorldSplat is not 3D Gaussian Splatting

## From single-scene reconstruction to a learned scene manifold

**Technical note — August 2026**

> **Do not hype. Do not lie. Just show.**

WorldSplat uses splats and a perspective renderer, so at first glance it can look like a small version of **3D Gaussian Splatting (3DGS)**. That is not the right comparison.

The original 3DGS problem is approximately:

> **Given many calibrated views of one physical scene, optimize an explicit cloud of 3-D Gaussians that reconstructs that one scene and renders novel views well.**

WorldSplat is trying to solve a different problem:

> **Given many examples drawn from a class of scenes, learn one compact decoder that can emit an entire splatted scene hypothesis from a small latent state.**

The distinction is not "big model versus small model" and it is not "Gaussian splats versus some other splats." The main distinction is **what is being optimized and what is meant to persist after training**.

This document explains that difference, records how the project arrived here through the SplatWorld / TinyAvatar / SlapStack lineage, and states clearly what the current WorldSplat branch can and cannot claim.

---

## 1. The superficial similarity

Both systems contain a set of spatial primitives and a renderer.

A conventional 3DGS scene contains explicit 3-D Gaussians with learned scene-specific properties. The original method introduced an explicit Gaussian representation, anisotropic covariance optimization, adaptive density control, and a visibility-aware real-time splatting renderer for high-quality novel-view synthesis from calibrated multi-view captures.

WorldSplat also decodes a set of spatial primitives and projects them through a camera. In the current experimental branch each latent emits a fixed number of soft splats with approximately:

```text
position / ray
learned depth
size
RGB
opacity
```

That shared word — **splat** — hides the more important difference.

---

## 2. What original 3D Gaussian Splatting optimizes

Let a captured physical scene be `S`, with calibrated cameras `C_v` and images `I_v`.

A simplified view of scene-specific 3DGS training is:

```text
one captured scene
    +
many calibrated views
        ↓
optimize the explicit Gaussian set S
        ↓
render that same scene from new cameras
```

Mathematically, the optimization is roughly of the form

```text
S* = argmin_S Σ_v L( R(S, C_v), I_v )
```

where the optimized parameters belong to **this scene**.

Train on a garden and the Gaussian cloud represents that garden.
Train on a kitchen and a different optimization produces a different Gaussian cloud.

The learned artifact is the **scene itself**.

This is one reason 3DGS can reconstruct a single environment extraordinarily well: the representation is allowed to spend its parameters on the exact geometry, appearance and view-dependent radiance of that particular capture.

The original 3DGS paper by Kerbl et al. is therefore best understood here as a powerful **scene reconstruction / novel-view synthesis method**, not as the direct ancestor of the WorldSplat learning objective.

---

## 3. What WorldSplat optimizes instead

WorldSplat does not normally optimize one independent splat cloud per training image and keep those clouds.

It learns a shared mapping:

```text
image
  ↓
encoder
  ↓
small latent z
  ↓
shared decoder
  ↓
whole splat configuration
  ↓
renderer
```

The parameters that survive training are primarily the **encoder and decoder weights** — a function that maps points in latent space to scene hypotheses.

A simplified objective looks more like:

```text
θ*, φ* = argmin E_I [
    L( R( D_θ(E_φ(I)) ), I )
]
```

with auxiliary terms for depth, latent regularization, sparsity, etc.

The important object is therefore not one optimized scene `S*` but a learned function

```text
D_θ : z → S(z)
```

that can emit many different splatted configurations.

This is much closer to the old SplatWorld move:

```text
128-D latent
    ↓
shared decoder
    ↓
256 coordinated packets
    ↓
face
```

except that WorldSplat is trying to make the decoded object geometric rather than only image-plane structured.

### The optimization axis is different

A concise way to say it is:

```text
3DGS:
    optimize parameters ACROSS VIEWS of ONE scene

WorldSplat:
    optimize a decoder ACROSS MANY scenes / observations
```

The ambition is not merely to reconstruct one street better than 3DGS.

It is to learn a **small reusable scene prior** that can rapidly instantiate a new scene hypothesis.

---

## 4. Side-by-side

| | Original 3D Gaussian Splatting | WorldSplat, current line |
|---|---|---|
| Main target | One captured scene | A distribution / manifold of scene hypotheses |
| Training unit | Multi-view observations of the same scene | Many training examples sharing one encoder/decoder |
| Parameters after fitting | Explicit scene-specific Gaussians | Shared network weights + per-example latent `z` |
| New scene | Usually optimize a new representation | Encode / infer a new latent, then decode |
| Number of primitives | Adaptive scene-specific cloud | Fixed compact splat budget in current v0 |
| Gaussian shape | Original 3DGS uses anisotropic covariance | Current WorldSplat uses deliberately simple soft/isotropic splats |
| Appearance | Scene-specific radiance, including view dependence in 3DGS | Current WorldSplat predicts simple RGB + opacity |
| Renderer | Highly optimized real-time splatting | Small inspectable PyTorch research renderer |
| Strength | High-fidelity reconstruction of a captured scene | Compact amortized generative scene hypothesis |
| Current weakness | Must fit/capture each scene | Geometry, latent prior, binding and persistence are still research problems |
| Sampling / morphing | Not the central original objective | Central diagnostic: random worlds, encoded worlds, latent morphs |
| "World" meaning | The fitted Gaussian cloud *is that captured world* | The decoder contains regularities across possible worlds; a latent instantiates one hypothesis |

This table should prevent the most tempting bad claim:

> **WorldSplat is not a smaller or better 3DGS.**

It is using a splatted renderer inside a different learning problem.

---

## 5. Why this distinction matters for the VKITTI2 experiment

The current Virtual KITTI 2 branch trains one shared WorldSplat model over thousands of RGB + metric-depth examples.

It does **not** give each street frame an independent large Gaussian cloud and optimize that cloud until it perfectly reproduces the image.

Instead every frame must pass through the same bottleneck:

```text
RGB frame
   ↓
encoder
   ↓
96-D latent
   ↓
shared decoder
   ↓
512 splats
```

in the strict Phase-1 A/B configuration.

That is a severe compression and sharing constraint. The decoder is being asked to discover recurring regularities such as, potentially:

```text
road-like surfaces
horizon structure
near / far organization
vertical masses
vegetation statistics
vehicle-like local structure
sky / far field
```

without allocating a completely independent parameter cloud to every frame.

If it eventually works, the interesting fact will not be that splats can reconstruct a road scene — 3DGS and many other methods already establish that explicit scene representations can do that very well.

The interesting fact would be that a **small shared decoder learned a useful geometry of road-world hypotheses across many scenes**.

That is a different question.

---

## 6. The current Phase-1 ray fix

The first VKITTI2 run exposed an important inherited mistake.

SplatWorld and TinyAvatar used packet anchors that behaved like **image-plane positions**. In the first WorldSplat 3-D conversion those bounded anchors were reinterpreted as world-space `x,y`, while a learned `z` was added and perspective projection was applied:

```text
x,y = image-like anchor + offset
z   = depth
u   = f*x/z
v   = f*y/z
```

This made far splats lose access to the image periphery. Metric depth asked distant buildings and sky to move outward in `z`, while RGB reconstruction still needed them at the image edge. The parameterization could not satisfy both constraints.

The current `agent/worldsplat-vkitti2-rayfix` branch restores the old semantic meaning of the anchor:

```text
u,v = image-plane ray anchor + offset
z   = learned depth
x,y = (u,v) * z / f
```

so at zero camera rotation:

```text
f*x/z = u
f*y/z = v
```

exactly.

This is deliberately a **coordinate correction**, not a new world-model claim.

The old 80k run remains the control.

---

## 7. Why CelebA made this family look smarter than it was

The early SplatWorld/TinyAvatar systems benefited from a hidden gift: **CelebA is strongly canonicalized**.

Across roughly two hundred thousand face images,

```text
eyes  → usually near the same region
nose  → usually near the same region
mouth → usually near the same region
head  → usually centered
```

A shared fixed packet field can therefore acquire stable roles across the dataset.

The system was never told "this packet is an eyebrow." Yet repeated reconstruction pressure can make coordinated groups of packets consistently participate in eyebrow-like, cheek-like or mouth-like structure.

That is why the face manifold could emerge without a literal face graph in the code.

The first RGB-only WorldSplat face run then added learned depth with no depth teacher at all. It produced a rotatable but concave / hollow face. This was not correct geometry, but it was informative: the shared decoder had organized its primitives into a coherent renderable relief because doing so helped explain a highly regular class of images.

VKITTI removes that easy canonical frame.

A tree can be left, right, near, far or gone. A car moves relative to the camera. Buildings sweep across the image. The dataset no longer supplies a single object-centered coordinate system for free.

That is why the transition from faces to worlds is not just "use more splats."

---

## 8. The repo lineage: how we got here

WorldSplat did not begin from 3DGS and then get compressed. It arrived through a different experimental lineage.

### 8.1 SlapStack — splats are not automatically a world

The SlapStack series explored atom / packet representations inside a layered interactive scene. Later versions explicitly had concepts above the primitive level: assignments, object templates, pose, depth, occlusion and binding.

The important inherited lesson is:

> **A soup of primitives is not yet a scene model. The world appears when primitives participate in persistent higher-level relations.**

This becomes important again now that WorldSplat can render 3-D-ish primitives but still lacks full persistent scene binding.

Repo: `anttiluode/SlapStack9`

### 8.2 SplatWorld — one small decoder can contain a family, not one image

SplatWorld trained a compact VAE on CelebA. Its decoder mapped a latent vector into hundreds of Gabor packets whose interference became the image.

The decisive conceptual move was not the Gabor basis itself. It was:

> **A tiny coordinate `z` can invoke a large, coordinated structured hypothesis because the shared decoder has learned the regularities of an entire dataset.**

The model did not store one face. It represented a **face manifold**.

The strange "fire" between well-inhabited latent regions also became diagnostic: outside the coordinated manifold the raw primitives became visible as soup/interference.

Repo: `anttiluode/SplatWorld`

### 8.3 TinyAvatar — acquisition versus motion inside the manifold

TinyAvatar reused the learned packet manifold as a live avatar substrate. The encoder could acquire a state from an image; subsequent motion experiments explored how to move through the learned manifold rather than repaint every frame from scratch.

This pushed the project from "small generative picture" toward "small internal state that can be driven."

Repo: `anttiluode/TinyAvatar`

### 8.4 TinyAvatar2 — the decoder itself became an implicit rig

TinyAvatar2 made the manifold mechanically inspectable. The decoder Jacobian could be used to ask:

```text
if I pull this rendered region,
which latent directions move it,
and what else moves with it?
```

Experiments found structured stiffness / compliance and coordinated motion in trained models. The face was not encoded by a hand-built skeleton; the collective decoder geometry acted like an implicit rig.

The inherited lesson for WorldModel is profound but narrow:

> **Useful object structure can live in the geometry of a learned decoder even when no explicit symbolic object graph was supplied.**

Repo: `anttiluode/TinyAvatar2`

### 8.5 TheSplat5 — a rich prior can be held by sparse evidence, and can also hallucinate

TheSplat line separated **acquisition** from **holding**. A full encoder could place the system into the correct latent basin, while sparse motion / feature observations could keep the belief attached to the live stream.

It also exposed the frontal-face attractor: point the camera at something that is not a face and the strong face manifold can still project a face-like belief.

That lesson later became central to WorldModel:

> **A rich prior is useful, but prior strength is not the same as evidence that the current world actually contains what the prior predicts.**

Repo: `anttiluode/TheSplat5`

### 8.6 SplatField — the learned data world and the dynamical field world are different objects

SplatField froze a trained basis and gave its coefficients their own recurrent dynamics. This produced the explicit "two worlds" observation:

- the **data world** shaped by the training distribution;
- the **field / Gram world** shaped by interactions among the basis functions.

The broader architectural lesson is separation of roles. A representation learned from data and the dynamics operating on that representation need not obey the same geometry.

Repo: `anttiluode/SplatField`

### 8.7 SplatNeuron and SplatNeuronPlusField — compact observers, external innovation and recursive projection

The later SplatNeuron work increasingly attacked the assumption that compact structured representations were automatically special. Strong generic baselines, byte accounting and failure ledgers became part of the method.

SplatNeuronPlusField sharpened another distinction that fed directly into WorldModel:

```text
external innovation
    versus
recursive projection of the system's own descendants
```

A repeated prediction can look like another observation while adding no independent information.

Repos:

- `anttiluode/SplatNeuron`
- `anttiluode/SplatNeuronPlusField`

### 8.8 WorldModel — content, support and lineage

The anchored-world branch therefore starts from a state larger than merely "what scene do I believe?"

```text
World = (content, external support, lineage)
```

The generative representation is allowed to complete unseen structure, but completion should not automatically become evidence.

This is orthogonal to the choice of splat renderer. It is a bookkeeping and active-observation problem that becomes more important as the scene prior gets stronger.

Repo: `anttiluode/WorldModel`

### 8.9 WorldSplat — lifting the learned manifold from an object class toward scene space

WorldSplat combines the SplatWorld/TinyAvatar idea of a shared compact decoder with an explicit 3-D-ish rendering substrate.

The question became:

> **Can a small decoder learn not one reconstructed environment, but a reusable geometry of possible environments?**

The hollow CelebA face and the first failed VKITTI2 geometry run are not side stories. They are the first two diagnostics of that exact question.

---

## 9. What is still missing: the actual persistent world layer

Current WorldSplat is still closer to:

```text
frame_t → z_t → scene hypothesis_t
frame_u → z_u → scene hypothesis_u
```

than to:

```text
                         one persistent W
                      /       |        \
                 camera_t camera_u camera_v
                    ↓        ↓        ↓
                  frame_t  frame_u  frame_v
```

That distinction matters more than the choice of splat primitive.

A true multi-view world state should be constrained by several observations of the **same external geometry**.

This is where Virtual KITTI 2 becomes especially useful: it supplies stereo cameras, camera calibration / poses, depth and repeated trajectories.

---

## 10. Why the two VKITTI cameras may be the important next experiment

The current branch deliberately uses Camera_0 only.

Simply adding Camera_1 images to the training folder as unrelated examples would not create stereo geometry. The learning objective must explicitly tell the model that the two images are observations of the **same scene state from two known cameras**.

A minimal next model is:

```text
Camera_0 image ─┐
                ├──> shared latent / shared decoded scene S
Camera_1 image ─┘

S + camera_0 → render_0 → compare to image_0
S + camera_1 → render_1 → compare to image_1
```

or, even more cleanly during training,

```text
z_scene
   ↓
shared decoder
   ↓
3-D scene S
   ├── render with C0 → I0
   └── render with C1 → I1
```

The loss then becomes approximately:

```text
L = L_rgb(R(S,C0), I0)
  + L_rgb(R(S,C1), I1)
```

with optional depth terms.

Now an arbitrary shallow painted relief cannot satisfy both views unless its geometry is at least compatible with the stereo baseline.

This is the first experiment in the lineage that would begin to ask for **one small learned scene representation that must survive genuinely distinct observations of the same world**.

That is substantially closer to the WorldModel goal.

---

## 11. The long-term distinction from 3DGS becomes even clearer there

A future WorldSplat does not need to compete with 3DGS at the task 3DGS was designed to dominate.

A mature division of labor could be:

```text
3DGS-style explicit geometry
    = excellent token-level representation of THIS captured scene

WorldSplat-style learned decoder
    = compact type-level prior over what scenes tend to be like

Anchored WorldModel
    = tracks which parts of THIS scene are observed,
      inferred, stale, or recursively generated
```

Those can even coexist.

A scene-specific Gaussian map could be the high-fidelity external memory while a small learned prior proposes structure in holes, predicts dynamics, compresses familiar regions, or supplies hypotheses when evidence is sparse.

The contribution, if one eventually exists, would therefore not be "replace Gaussian Splatting."

It would be closer to:

> **Put a learned generative manifold, a persistent metric scene, and explicit evidential support into one system without allowing the prior to counterfeit observation.**

That remains a research target, not a result.

---

## 12. Claims and non-claims

### What is shown so far

- A compact SplatWorld-style decoder can learn a coordinated manifold over a highly regular image class.
- An RGB-only 3-D soft-splat extension trained on CelebA produced a manipulable / rotatable pseudo-3-D face representation without any external depth supervision.
- Its geometry was wrong — notably hollow/concave — which demonstrates monocular underconstraint rather than correct 3-D recovery.
- The first metric-depth VKITTI2 run learned coarse road/tree/sky-like structure but exposed a concrete coordinate conflict inherited from image-plane anchors.
- The Phase-1 branch now tests the minimal ray-coordinate correction while preserving the failed run as control.

### What is not shown

- WorldSplat has not beaten 3DGS on scene reconstruction.
- It is not currently a production Gaussian-splat renderer.
- It has not yet demonstrated accurate stereo geometry.
- It has not yet demonstrated a persistent world shared across time/views.
- It has not yet demonstrated useful semantic objects or physical dynamics.
- The current work does not establish novelty over the broad literature on generalizable NeRFs, feed-forward Gaussian reconstruction, generative 3-D models, or amortized scene representations.
- A pretty novel-view render is not by itself evidence of a correct world model.

---

## 13. The experimental ladder from here

The clean sequence is:

```text
A. completed VKITTI2 control
   bounded world-x/y + depth
   → peripheral geometry conflict

B. Phase-1 ray fix
   ray position independent of depth
   → does the fog / central collapse improve?

C. Stereo
   one decoded scene must explain Camera_0 and Camera_1
   → geometry without relying on supplied depth

D. Multi-frame shared world
   one state must explain several poses along a trajectory

E. Persistent update
   previous world + new observation → updated world

F. Anchored completion
   learned prior fills unseen regions
   while support/lineage still says what was actually observed
```

Every step has an attacker and a stop line.

If the ray fix does not materially improve the VKITTI failure, do not rescue it by silently increasing capacity.

If stereo cannot beat a flat / depth-image / conventional geometry baseline, do not call it learned 3-D.

If a standard scene-specific Gaussian map does the task better and no cross-scene prior is required, use the standard map.

---

## 14. One-sentence distinction

If someone asks what the project is doing that ordinary 3D Gaussian Splatting is not doing, the shortest honest answer is:

> **3DGS fits an explicit splat cloud to one observed scene; WorldSplat is trying to learn a small shared function that maps a latent state to an entire scene hypothesis, so the regularities of many worlds live in the decoder rather than in one fitted cloud.**

And the longer-term WorldModel adds one more clause:

> **...while separately remembering which parts of the instantiated world are supported by external observation and which parts came from the prior.**

---

## References / project lineage

### 3D Gaussian Splatting

- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis. **3D Gaussian Splatting for Real-Time Radiance Field Rendering.** ACM Transactions on Graphics / SIGGRAPH 2023. Project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Official implementation: https://github.com/graphdeco-inria/gaussian-splatting

### This project lineage

- SplatWorld: https://github.com/anttiluode/SplatWorld
- TinyAvatar: https://github.com/anttiluode/TinyAvatar
- TinyAvatar2: https://github.com/anttiluode/TinyAvatar2
- SlapStack9: https://github.com/anttiluode/SlapStack9
- TheSplat5: https://github.com/anttiluode/TheSplat5
- SplatField: https://github.com/anttiluode/SplatField
- SplatNeuron: https://github.com/anttiluode/SplatNeuron
- SplatNeuronPlusField: https://github.com/anttiluode/SplatNeuronPlusField
- WorldModel: https://github.com/anttiluode/WorldModel

---

## Status

This is a technical note documenting the design distinction and experimental lineage. It is **not a peer-reviewed paper and not a novelty claim**. The most important empirical question at the time of writing is whether the Phase-1 ray-coordinate fix materially improves the failed VKITTI2 geometry control. Stereo / shared-view training comes after that result, not before it.
