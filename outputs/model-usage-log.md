# Model Usage Log

## Current Session

| Work Type | Model / Method | Why |
| --- | --- | --- |
| File inventory, section checks and stale-claim scans | Deterministic tools/scripts | Avoid guessing project state and detect old empirical claims. |
| Expanded COCO experiment | Deterministic Python/PyTorch execution | All metrics come from saved predictions, annotations and CSV outputs. |
| Size-metric correction and robustness baseline | Deterministic post-processing script | Prevent misleading size precision and compare perturbations against original images. |
| Dissertation rewriting and synthesis | High-reasoning model tier | Needed for Level 7 academic structure, critical interpretation and evidence-to-robotics discussion. |
| DOCX generation | Python-docx plus Microsoft Word field update | Duplicates the UCA working template copy and fits revised content into DOCX. |
| QA checks | Deterministic scan, fast QA, a11y audit, Word PDF export and Poppler render | Validates structure, word count, accessibility, stale terms and page rendering without invented judgement. |

Exact API model ID was not recorded by the local project runner. Future API-based runs should log the exact model ID when the execution environment exposes it.
