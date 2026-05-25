# Layer 4 · Biometric Matching

> **File:** [`layer4_biometrics.py`](../layer4_biometrics.py)
> **Goal:** Once upstream layers have established the input is a live
> human, decide whether that human is **the same person** as a stored
> reference ID. Powered by InsightFace's ArcFace recognition pipeline.

---

## 1. What it does

```
reference ID                                         probe (live capture)
     │                                                       │
     ▼                                                       ▼
EXIF transpose                                       EXIF transpose
     │                                                       │
     ▼                                                       ▼
InsightFace FaceAnalysis pipeline                  InsightFace FaceAnalysis
     │   ┌─ RetinaFace detector (det_10g.onnx)              │
     │   ├─ landmark alignment to 112×112                   │
     │   └─ ArcFace W600K-R50 (w600k_r50.onnx)              │
     ▼                                                       ▼
   embedding_ref  (512-D, L2-normalised)            embedding_probe
                       │                                  │
                       └────────── dot product ───────────┘
                                       │
                                       ▼
                          cosine similarity ∈ [−1, +1]
                                       │
                                       ▼
                        sim ≥ threshold  →  MATCH
                        sim <  threshold  →  NO MATCH
```

### Key design choices

- **InsightFace `buffalo_l` bundle.** Standard CPU-friendly recipe:
  RetinaFace detector + ArcFace W600K-R50 backbone, plus age/gender
  and 2D/3D landmark sub-models (not used in this layer but bundled
  for free). Auto-downloaded to `~/.insightface/models/` on first
  use (~280 MB).
- **CPU execution.** `providers=['CPUExecutionProvider']` pins ONNX
  Runtime to CPU for predictable cross-machine behaviour. ~50–200 ms
  per face on an M3 laptop. For GPU, prepend `CUDAExecutionProvider`.
- **L2-normalised embeddings.** ArcFace's training objective places
  embeddings on the unit hypersphere by construction, so cosine
  similarity reduces to a plain dot product — no extra normalisation
  required.
- **Largest face only.** Both reference and probe pictures are
  assumed to contain a single subject; we pick the largest detection
  if multiple fire.

### What a "good" similarity looks like

| Pair                                          | Typical cosine sim. | Verdict @ 0.40 |
|-----------------------------------------------|---------------------|----------------|
| Same shot, same image                         | **1.000**           | MATCH          |
| Same person, different lighting/angle         | 0.55 – 0.85         | MATCH          |
| Same person, 5+ years of ageing               | 0.40 – 0.55         | borderline     |
| Same person, heavy occlusion (mask, sunglasses) | 0.25 – 0.45       | borderline     |
| Different person, generic comparison          | −0.05 – 0.25        | NO MATCH       |
| Different person, similar facial features     | 0.15 – 0.35         | NO MATCH       |

### Threshold semantics

| threshold | character            | typical use                          |
|-----------|----------------------|--------------------------------------|
| 0.30      | very loose           | risk-tolerant onboarding             |
| **0.40**  | **balanced**         | **default — KYC, login**             |
| 0.50      | strict               | high-stakes auth, financial actions  |
| 0.60      | very strict          | duplicate detection across a DB      |

The default 0.40 corresponds to the standard balanced operating point
on LFW, CFP-FP, and AgeDB-30 benchmarks. Tune against a labelled
real-world pair dataset in production — the optimal threshold depends
heavily on demographic mix, capture conditions, and the cost ratio
between false accepts and false rejects.

---

## 2. Pros

- **Best-in-class accuracy from classical CNN methods.** ArcFace +
  W600K-R50 hits ~99.8 % on LFW, ~98 % on CFP-FP, ~97 % on AgeDB-30.
  Strong across ethnicity, age, pose, illumination compared to
  earlier baselines.
- **No training data required.** Pre-trained embeddings; you only
  pick the threshold.
- **Pure dot-product matching.** Trivially extends to a 1-to-N
  search by storing the embeddings in any vector DB (FAISS,
  Pinecone, plain numpy) and computing similarity in O(N·D).
- **CPU real-time** — ~50–200 ms per face on a modern laptop, no
  GPU required.
- **Deterministic** — the same image always produces the same
  embedding; the threshold is the only judgement call.
- **Explainable to operators** — a single similarity score with a
  clear threshold is auditable in a way a deep PAD score isn't.
- **Embedding diagnostics in the UI** — the first 32 dims and the
  element-wise product are shown so a reviewer can see *which*
  dimensions are pulling the score up or down.

## 3. Cons & failure modes

- **The model is opaque about *why* two faces match.** A high
  similarity says "the embedding network put these on the same
  region of the hypersphere"; it doesn't tell you which features
  drove the decision. Hard to audit at a per-claim level.
- **Demographic bias.** ArcFace, like most face recognition models,
  has well-documented disparities across race, age, and gender —
  generally higher false-reject rates for women and for darker
  skin tones, especially under poor illumination. Re-calibrate
  thresholds per cohort or use cohort-aware models.
- **Ageing degrades similarity.** Photos > 5 years apart drop into
  the 0.4–0.55 range even for the same person. Adjust threshold
  or accept that very old reference photos will be flaky.
- **Heavy occlusion fails predictably.** Masks, sunglasses, very
  off-angle poses, and partial occlusion all suppress similarity.
- **Spoofs trivially pass this layer** — a printed photo or screen
  replay of the reference image produces an embedding nearly
  identical to the reference's. This layer **must** be gated by
  Layers 1–3 (injection / deepfake / passive PAD) or it provides
  zero security against simple presentation attacks.
- **InsightFace must download ~280 MB on first run.** First-launch
  UX is bad without pre-staging. Subsequent loads take ~2 s.
- **Single-face assumption.** Group photos and crowded scenes lose
  information; we pick the largest face and discard the rest.
- **ONNX Runtime on macOS occasionally exhibits subtle numerical
  differences from Linux CUDA — embeddings are slightly different
  bit-for-bit.** Cosine similarity differences are below 1 %, but
  if you cache embeddings on one platform and compare on another
  the threshold operating point may drift slightly.

---

## 4. Alternatives & complementary recognisers

### 4.1 FaceNet (OpenFace, FaceNet-PyTorch)
Older but still respectable triplet-loss embeddings.
- **Pros:** Smaller models, well-documented, many tutorials.
- **Cons:** Significantly worse accuracy than ArcFace, especially
  on hard cohorts.

### 4.2 AdaFace / MagFace / CurricularFace
Newer ArcFace-family losses that improve robustness to low-quality
inputs (blurry, low-light, low-resolution).
- **Pros:** Higher accuracy on hard cases; AdaFace gracefully
  degrades on poor inputs.
- **Cons:** Same opacity and bias issues; same compute footprint.

### 4.3 Foundation-model face encoders (e.g. DINOv2-finetune)
Vision-foundation-model embeddings fine-tuned on faces.
- **Pros:** Strong generalisation across domains; benefits from
  every new foundation-model release.
- **Cons:** Heavier inference; opacity remains; less mature
  tooling.

### 4.4 Commercial APIs (AWS Rekognition, Azure Face, Paravision)
- **Pros:** Outsource the engineering; well-calibrated thresholds;
  geographic compliance helpers (PII handling).
- **Cons:** Per-request cost, latency, vendor lock-in,
  cross-border data-residency complications.

### 4.5 1-to-N search with vector DB
Same embedding pipeline but matching against a database, not a
single reference (deduplication, watchlists).
- **Pros:** Trivially extends — InsightFace embeddings work with
  any vector DB.
- **Cons:** False-accept rate grows linearly with database size;
  threshold must be tightened. At 1M entries you typically need
  a threshold of 0.55–0.60.

### 4.6 Multi-modal verification (face + voice + behaviour)
- **Pros:** Compounds with face — independent failure modes give
  geometric reduction in attack success rates.
- **Cons:** Each modality adds UX friction.

---

## 5. How this layer fits into the full stack

| Layer | Catches                                  | Cost           |
| ----- | ---------------------------------------- | -------------- |
| 1     | Injection (virtual cams, file replay)    | µs/frame       |
| 2     | Deepfake / screen / print (YOLO + FFT)   | ~30 ms/frame   |
| 3     | Print / screen physical texture (LBP)    | ~10 ms/frame   |
| **4** | **Identity match (InsightFace embeddings)** | **~100 ms/face** |

Layer 4 is the most expensive layer **and** the only one that
answers the identity question. It also has the most catastrophic
failure mode: **it cannot tell a real face from a photo of the same
face**. The entire security model depends on Layers 1–3 having
already excluded spoof inputs before we run the recogniser.

The expected order of operations in a production gate:

1. Layer 1 confirms the stream is from real hardware (kills bots).
2. Layer 2 confirms the image isn't a deepfake or screen replay
   (kills digital spoofs).
3. Layer 3 confirms the texture is live human skin (kills physical
   spoofs).
4. **Only then** does Layer 4 confirm the identity matches the
   reference ID.

Running Layer 4 first — to "save time on obvious matches" — is the
classic mistake that lets attackers bypass biometric gates with a
printed photo.

---

## 6. Testing recipes

1. **Same image twice** (upload identical files) → sim = 1.000.
   Sanity check that the pipeline is wired up correctly.

2. **Two real photos of yourself, different days / lighting** →
   expect sim 0.6–0.9 → MATCH at any reasonable threshold.

3. **You vs a colleague** → expect sim −0.05 to 0.25 → NO MATCH
   comfortably below threshold.

4. **Photo of you at 18 vs photo at 35** → sim drops to 0.4–0.6.
   May land borderline at threshold 0.50; passes at 0.40.

5. **Same person but one frame heavily occluded** (sunglasses,
   mask, hat) → sim drops to 0.25–0.50 depending on how much skin
   is visible.

6. **Reference photo vs your phone showing the reference photo →
   capture with webcam** → similarity ≈ 0.95+. **MATCH** —
   correctly demonstrating that this layer alone gives a printed
   spoof full credit. Layers 1–3 must catch this.

If a legitimate self-pair scores below threshold, look at:
- **Reference detection confidence** — < 0.5 means the ID image is
  too low-quality to embed reliably.
- **Probe detection confidence** — same problem on the probe side.
- The **5 keypoints** drawn on each face. If they're miss-aligned
  (e.g. on the ear, not the eye), the alignment step is corrupting
  the crop and the embedding is meaningless.
- The **embedding diagnostics expander** — if very few dimensions
  contribute positively, one of the faces is likely off-pose or
  poorly cropped.

---

## 7. Recommendations for production

1. **Pre-stage the model** in the container image or on first install
   — don't make production users wait for a 280 MB download.
2. **Calibrate the threshold against a labelled dataset** that
   reflects your real user demographic. The default 0.40 is a
   starting point, not a final number.
3. **Monitor demographic equity.** Track false-accept and
   false-reject rates by ethnicity / age / gender; ArcFace has
   known disparities and you need to see them in your data, not
   trust them to be benign.
4. **Use 1-to-1 verification, not 1-to-N search**, unless your
   threat model specifically requires watchlist matching. 1-to-N
   compounds the false-accept rate with the database size.
5. **Always gate this layer behind PAD layers.** It has no
   defence against printed photos.
6. **Cache reference embeddings**, not images. The ID photo gets
   embedded once at enrolment; subsequent verifications only need
   to embed the probe (~100 ms instead of ~200 ms).
7. **Log per-claim**: similarity score, threshold, ref-image
   provenance, probe-image provenance, model version. The
   recogniser is opaque; the audit trail must not be.
8. **Have an escalation path.** Borderline scores (e.g. 0.30–0.45)
   should be routed to a human reviewer, not auto-rejected.
