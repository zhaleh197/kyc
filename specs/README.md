# Specs

One spec per module, plus one for what every module shares and one for the
orchestrator that combines them.

| Spec | Module | Kind | Status |
|---|---|---|---|
| [00-platform.md](00-platform.md) | shared contract, config, security | reference | describes built code |
| [01-document-ocr.md](01-document-ocr.md) | Document OCR | **retrospective** | built, tested |
| [02-face-quality.md](02-face-quality.md) | Face quality / selfie capture | up-front | prototype in `phase1_facedetect/` |
| [03-face-match.md](03-face-match.md) | Face matching | up-front | prototype in `phase2_facematch/` |
| [04-anti-spoof.md](04-anti-spoof.md) | Passive anti-spoofing | up-front | prototype in `phase3-antispoof/` |
| [05-liveness.md](05-liveness.md) | Active liveness | up-front | prototype in `phase4-liveness/` |
| [06-ai-forensics.md](06-ai-forensics.md) | Document & selfie forensics | up-front | not started |
| [07-orchestrator.md](07-orchestrator.md) | End-to-end verification | up-front | not started |

Suggested build order: 02 → 03 → 04 → 05 → 07 → 06. Face quality gates the
input of 03/04/05; face match is the first module the orchestrator actually
needs (portrait from OCR vs selfie); forensics is last because it is the one
with no working baseline to measure against.

## Why specs, and why these two kinds

**Retrospective (OCR)** — consolidates what was scattered across
`models/manifest.yaml`, `training/kaggle/README.md`, commit messages and
conversation history into one place, so the next module starts from what OCR
actually taught rather than from memory.

**Up-front (everything else)** — written before any migration code. OCR paid
for several assumptions that were never written down: Roboflow's export API
returning links to missing objects, Kaggle GPU mismatches, EasyOCR's `(N,T,C)`
vs this project's `(T,N,C)` CTC layout, a Windows DataLoader hanging for 12h+,
RTL vs LTR join order, a sharpness threshold calibrated for a different
geometry. Each up-front spec therefore has an **"Assumptions to verify before
building"** section: things to check in an hour now instead of discovering in
a week later.

## Living documents

A spec is not a contract. When real testing contradicts it, update the spec in
the same commit as the code change and say why. A spec that disagrees with the
code is worse than no spec.

Numbers are marked one of three ways:

- **measured** — has a source (manifest entry, test, eval run). Quote the source.
- **proposed** — a target chosen before measurement. Must be confirmed or
  revised once a baseline exists; never ship a proposed number as a threshold
  without measuring it first.
- **TBD** — nobody knows yet; the spec says how to find out.

## Template for a module spec

1. Purpose — one paragraph, what question the module answers
2. Scope / non-goals
3. Interfaces — endpoints, request, `data` payload, reason codes
4. Decision rules — what makes `pass` / `review` / `fail`
5. Models — which, from where, license, how served on CPU
6. Configuration — every tunable, with its env var
7. Acceptance criteria — measurable, each marked measured / proposed / TBD
8. Assumptions to verify before building
9. Migration from the prototype — keep / drop / rewrite
10. Test plan
11. Open questions
