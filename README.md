# sklearn Model Training + Evaluation

**Track:** Module 1 — ML & neural networks · **When:** Week 1, Day 4 · **Status:** Required

This repository contains everything you need for this lab.

## Files

- [`instructions.md`](./instructions.md) — the lab instructions
- [`rubric.md`](./rubric.md) — how your submission is graded; this is what the AI reviewer checks your PR against

## How to complete this lab

1. **Fork** this repository.
2. Do the work described in `instructions.md`, committing to your fork.
3. Open a **pull request** back into this repository.
4. You'll receive **AI feedback** on your PR based on `rubric.md`. Address any blocking feedback and push updates to the same PR.

## Submission hygiene

- Keep this repository scoped to this lab only — no unrelated projects or personal files.
- Use clear, descriptive filenames.
- Remove secrets, API keys, and tokens before committing.

## Analysis and Recommendations

The dataset contains 7,043 customers, and 26.54% of them churned. Churn is more common among customers with month-to-month contracts than among customers with longer contracts.

The best result among the tested KNN values was k=11: 77.36% accuracy, 57.70% precision for churn, and 55.08% recall for churn. The confusion matrix shows that the model correctly identified 206 of 374 customers who churned, but missed 168.

The model's accuracy is about 3.9 percentage points above a baseline that always predicts “No churn” (73.46%). This is a modest improvement, and the 55.08% churn recall means the model misses about 45% of churn cases.

The business should prioritize retention efforts for month-to-month customers. Before relying on this model for decisions, the team should try ways to detect more churn cases and evaluate the trade-off between catching more customers and generating false alarms.