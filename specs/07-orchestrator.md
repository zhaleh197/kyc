# 07 — Orchestrator (end-to-end verification)

Status: **in progress** (branch `orchestrator`, 2026-10-17): `kyc/modules/orchestrator/`, see §11. Originally an up-front spec, not started. Every module README/docstring already
assumes it exists ("the orchestrator aggregates them into one verdict").
`phase1_facedetect/claude_api/main.py` has a `/kyc/full` endpoint with a
weighted-sum score (e.g. forensics weight 0.20) — the shape to avoid, see §4.

## 1. Purpose

Run one customer's verification as a **case**: collect the card image and a
live selfie session, call the modules in order, combine their `ModuleResult`s
into a final `approved` / `manual_review` / `rejected`, and keep an audit
record an operator (and later an auditor) can read.

## 2. Flow

```
create case
  ├─ card image ─► 01 OCR ─► (06a document forensics)
  │                  └─ portrait
  ├─ camera session:
  │     02 face capture ─► selfie (capture_id)
  │       └─► 05 liveness on the same stream (identity = selfie, 04 on frames)
  └─ 03 face match(portrait, selfie)  +  04 on the selfie
        ─► policy ─► verdict + audit record
```

The selfie is taken by 02 and held server-side; it is never uploaded by the
client. 05 runs straight after on the same stream and checks every sampled
frame is the same person as that selfie, so the selfie matched to the card is
provably the live person.

## 3. Interfaces

| Method | Path | |
|---|---|---|
| POST | `/v1/cases` | create; returns `case_id` |
| POST | `/v1/cases/{id}/document` | card image → runs 01 (+06a) |
| POST | `/v1/cases/{id}/capture` | starts the 02 capture session bound to the case |
| POST | `/v1/cases/{id}/liveness` | starts 05 from the case's `capture_id` |
| POST | `/v1/cases/{id}/finalize` | runs 03, applies policy, returns verdict |
| GET | `/v1/cases/{id}` | verdict + every module result (for operators) |

Verdict payload: `verdict`, `policy_version`, `module_results[]`,
`blocking_reasons[]` (the reasons that drove the verdict), `attempts`.

## 4. Policy (combining results)

Rule-based, not a weighted sum:

- any module `fail` → `rejected` **or** `manual_review`, per a configurable
  map of reason code → outcome (e.g. `national_id_checksum` → review, since it
  may be a misread; `spoof_detected` → rejected)
- any module `review` → `manual_review`
- all `pass` → `approved`
- a module that raised (503) → `manual_review` with `module_unavailable`,
  never `approved`
- `under_age` (warn from 01) → per business rule

A weighted sum lets a strong OCR score compensate for a failed liveness —
exactly the wrong trade in KYC. Scores are for ranking the review queue, not
for the verdict.

The policy is versioned (`policy_version` on every verdict) so a stored
decision can be explained with the rules in force at the time.

## 5. State, attempts, retention

- Case state in a store (Redis/Postgres — decide), TTL for incomplete cases.
- Attempt limits per case: e.g. 3 document uploads, 3 liveness sessions
  (proposed) — unlimited retries turn every threshold into a brute-force
  target.
- **Retention**: images and embeddings are sensitive (biometric data +
  national id). Default: images discarded after the verdict, or kept
  encrypted for a fixed review window; the stored audit record holds module
  results with values masked where not needed. Retention period is a legal
  decision, not an engineering one — **must be set before production**.

## 6. Configuration — `KYC_ORCH_*`

`reason_outcome_map`, `max_document_attempts`, `max_liveness_attempts`,
`case_ttl_s`, `require_forensics` (off until 06 exists), `policy_version`.

## 7. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| O1 | No path yields `approved` if any module failed, errored, or was skipped | required |
| O2 | Every verdict lists the reason codes that determined it | required |
| O3 | A case can be re-evaluated against a newer policy from stored results without re-running models | proposed |
| O4 | End-to-end p95 (excluding user time in liveness) ≤ 5 s on CPU | proposed |
| O5 | Auth: a case is only readable by the tenant that created it | required |

## 8. Assumptions to verify before building

1. The OCR portrait crop is usable by 03 at all (03 §8.2).
2. A selfie captured from a 720p browser webcam is good enough for 03 against
   a small printed card portrait (measured as part of 03 M1).
3. Where manual review happens (existing back-office tool? new UI?) — decides
   the shape of `GET /v1/cases/{id}`.

## 9. Test plan

Policy unit tests from a table of (module results → expected verdict),
covering every O1 path. Integration: all modules with fake models, one full
case per verdict.

## 10. Open questions

- Multi-tenant from day one?
- Webhook to the client on verdict, or polling?
- Does the business need a registry check (06 §11) inside the case flow?

## 11. Build log (2026-10-17)

Built as specified in §3–§4: `POST /v1/cases`, `/document`, `/capture`,
`/liveness`, `/finalize`, `GET /v1/cases/{id}`. Policy in
`orchestrator/policy.py` as a pure function; service in
`orchestrator/service.py`; every module reached through its public pipeline
methods.

Decisions taken while building:

- **Server-held links.** The case starts the capture session, liveness starts
  from that capture's `capture_id`, face match uses the portrait OCR cropped
  from this case's card — the portrait never leaves the server and is dropped
  after the verdict. Nothing a client sends can substitute one of them.
- **Failure → outcome.** `reject_codes` (default: `spoof_detected`,
  `spoof_during_session`, `faces_do_not_match`, `face_changed`,
  `still_face_mismatch`, `card_expired`) reject; **any other failure goes to
  manual review** (a checksum failure may be a misread). Warnings review,
  unless listed in `reject_warning_codes` (business rule, e.g. `under_age`).
- **O1:** a step missing or a module unavailable (5xx) → manual review,
  never approved; a non-pass with no blocking reason still blocks.
- **Finalize refuses an incomplete case** (409 `case_incomplete`, listing the
  missing steps) rather than locking it with a "missing" verdict; once
  finalized the case is frozen and finalize is idempotent.
- **Attempts** per step (document / selfie / liveness, default 3) → 409
  `attempts_exhausted`. A new selfie clears the previous liveness session.
- Verdict carries `policy_version`, the blocking reasons with their outcome,
  per-step summaries, full module results without images, the OCR values for
  the operator, and `review_score` (min module score — ranking only).

End to end with the real models (dev machine, liveness off for the run):
the tester's card + selfie → **manual_review** (face 0.433 on a low-res
screenshot selfie, checksum misread, anti-spoof "retake"); another person's
card + the same selfie → **rejected** (`faces_do_not_match`, 0.138).

Open: auth / tenants (O5) — **blocker before any deployment**, as for every
endpoint; persistence (in memory today); retention period (legal); a
`retake_low_quality` from anti-spoof at finalize currently means manual
review — the client should instead be sent back to the selfie step before
finalizing.
