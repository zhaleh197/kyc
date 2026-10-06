# 07 — Orchestrator (end-to-end verification)

Status: **up-front spec, not started.** Every module README/docstring already
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
