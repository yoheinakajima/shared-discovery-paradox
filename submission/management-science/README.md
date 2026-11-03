# Shared Discovery — Management Science preparation

Source draft PR: https://github.com/yoheinakajima/shared-discovery-paradox/pull/5

This additive source package implements the reviewed first-target preparation at baseline `5025cc8e8f2f8ca015dff2066f08f81ad5715a51`. The original manuscript/PDFs, script, saved data, figures and history are preserved. **Prepared for review, not certified submission-ready.** Management Science Organizations remains the already-authorized first preparation target; there is no submission, acceptance, arXiv replacement, merge or deployment claim.

- Review pair: `manuscript.tex` / `manuscript.pdf`; `companion.tex` / `companion.pdf`; `bibliography.bib`; generated `main-labels.tex`.
- Science: `scientific-audit.md`, `reference-audit.md`, `reference-metadata.json`, `organizations-fit.md`.
- Reproducibility: `replication-guide.md`, exact `requirements-lock.txt`, `verification-receipt.json`, new native log, attributed `replication-package.zip` and its contents manifest, prepared for First Look/editor request rather than automatic initial upload.
- Named author assets (separate from blind review): `author-information.tex` / `author-information.pdf`, updated `author-declarations-template.md`, `tool-assistance.md` and public-source `tool-use-evidence.json`. Build the named page alone with `python3 submission/management-science/build.py --author-information-only`.
- Offline submission: `cover-letter-draft.md`, `ascollected-draft.md`, `author-declarations-template.md`, `reviewer-editor-worksheet.md`, `portal-checklist.md`.
- Handoff: `readiness.md`, `task-disposition.json`, `provenance.json`, `qa-receipt.json`.

Keep `author-information.pdf` separate from the anonymous main and electronic companion. The named page is an author-information candidate for later appropriate use, not an initial-review title-page upload. Unsubmitted status and remaining owner/portal gates are recorded in `readiness.md` and `portal-checklist.md`, outside candidate PDF bodies.

Build with existing cached Tectonic using `python3 submission/management-science/build.py`; no TeX download/install. Check frozen files/PDFs/metadata with `python3 submission/management-science/verify_package.py` (no numerical rerun/provider access). Read `replication-guide.md` before any optional later numerical replay; use a disposable output path. One full unchanged native verification route already passed for this preparation.

This is a **public repository**. Named assets contain author-approved public-ready facts; unresolved declarations remain explicit. Private owner answers, private conflicts, editable AsCollected URL, account details and editorial correspondence belong outside Git, or in the ignored `private-local/` directory. Atlas may hold status and source pointers, never those private answers.
