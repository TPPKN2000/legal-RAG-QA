# Design Experiment
## Test Dataset
`.\data\ALQAC2026_public_test.json`
## Submission Format
The submission must contain a list of predictions, one object per test case:
`submission_example.json`

A submission may be rejected or partially ignored if it violates the required format.

The organizers may validate the following conditions:

- Every test case has exactly one submitted prediction.

- Every `case_id` exists in the official test set.

- There are no duplicate `case_id`s.

- The `prediction` value is either A_WIN | B_WIN | PARTIAL_A_WIN | PARTIAL_B_WIN.

- `law_evidence` is a list of valid legal provision identifiers from the law corpus.

- The JSON file is valid and can be parsed automatically.

Duplicate evidence items may be automatically deduplicated before scoring.
### Required Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_id` | string | Yes | Public identifier of the test case. |
| `prediction` | string | Yes | Must be either `A_WIN`, `B_WIN`, `PARTIAL_A_WIN`, or `PARTIAL_B_WIN`. |
| `law_evidence` | list[object] | Yes | List of relevant legal provisions retrieved from the law corpus. Each object must contain `law_id` and `aid`. |

Each item in `law_evidence` must follow this format:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `law_id` | string | Yes | Identifier of the legal document in the law corpus. |
| `aid` | integer | Yes | Article/provision ID within the corresponding legal document. |
### Prediction labels
The `prediction` field must be one of the following values:

| Value | Definition |
|-------|------------|
| `A_WIN` | The court fully accepts all of the plaintiff's claims. |
| `PARTIAL_A_WIN` | The court partially accepts the plaintiff's claims, and the accepted portion is greater than 50%. |
| `PARTIAL_B_WIN` | The court partially accepts the plaintiff's claims, but the accepted portion is 50% or less. |
| `B_WIN` | The court fully rejects all of the plaintiff's claims. |

If a case contains multiple claims, teams should focus on the main claim described in the `case_query`.
## Evaluation Method
The official evaluation metrics for the Legal Case Outcome Prediction task.

The final score consists of three components:

- **Outcome Accuracy**: whether the system correctly predicts the winning side.
- **Penalized Case Evidence Recall**: whether the system retrieves the correct case-content evidence, with a penalty for excessive API calls.
- **Micro Law Evidence F1**: whether the system retrieves the correct legal provisions from the law corpus.

The final score is defined as:

\[
\text{FinalScore} = 0.70 \cdot \text{OutcomeAccuracy}
+ 0.20 \cdot \text{PenalizedCaseRecall}
+ 0.10 \cdot F1_{\text{micro}}^{\text{law}}
\]

The metric is designed to reward systems that:

a. Predict the correct outcome.

b. Retrieve the correct case evidence.

c. Retrieve the relevant legal provisions.

d. Use the Case Content API efficiently.
