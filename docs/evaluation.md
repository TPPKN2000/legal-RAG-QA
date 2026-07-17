# Evaluation Deep Dive
This is how the evaluation method works
## 2.1 Definitions

Let \(N\) be the number of test cases.

For each case \(i\):

| Symbol | Meaning |
|--------|---------|
| \(n_i\) | The total number of case-content segments available for case \(i\). |
| \(c_i\) | The number of Case Content API calls made by a system for case \(i\). |
| \(y_i\) | The gold outcome label for case \(i\). |
| \(\hat{y}_i\) | The predicted outcome label submitted by the system. |
| \(G_i^{\text{case}}\) | The gold set of case evidence segments for case \(i\). |
| \(P_i^{\text{case}}\) | The set of case evidence segments submitted by the system for case \(i\). |
| \(G^{\text{law}}\) | The set of all gold law evidence items across the full test set. |
| \(P^{\text{law}}\) | The set of all law evidence items submitted by the system across the full test set. |

## 2.2 Outcome Accuracy

For each case \(i\), the outcome score is defined as:

\[
\text{Outcome}_i =
\begin{cases}
1, & \text{if } \hat{y}_i = y_i \\
0, & \text{otherwise}
\end{cases}
\]

The overall outcome accuracy is:

\[
\text{OutcomeAccuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\text{Outcome}_i
\]

This component rewards systems that correctly predict whether the plaintiff or the defendant wins the case.

## 2.3 Case Evidence Recall

For each case, the system submits a set of case-content evidence segments.

The case evidence recall for case \(i\) is defined as:

\[
\text{Recall}_i^{\text{case}}
=
\frac{
\left| P_i^{\text{case}} \cap G_i^{\text{case}} \right|
}{
\left| G_i^{\text{case}} \right|
}
\]

This component measures how many gold case evidence segments are successfully retrieved by the system.

For example, if:

**Gold evidence**
= `{seg1, seg9, seg8}`

**Submitted evidence**
= `{seg1, seg2, seg8}`

Then:
**Corrected evidence**
= `{seg1, seg8}`

**Recall_case** = 2 / 3 = 0.667
## 2.4 API Efficiency Penalty

The API call budget is case-dependent. Larger cases have more segments, so they are allowed more API calls.

For case \(i\), the no-penalty API budget is:

\[
B_i = 2n_i
\]

This means that if case \(i\) has \(n_i\) segments, a system can make up to \(2n_i\) Case Content API calls without penalty.

The recall component starts being penalized only when:

\[
c_i > 2n_i
\]

The recall component becomes zero when:

\[
c_i \geq 5n_i
\]

The API efficiency factor is defined as:

\[
E_i =
\begin{cases}
1, & \text{if } c_i \leq 2n_i, \\
1 - \dfrac{c_i - 2n_i}{3n_i}, & \text{if } 2n_i < c_i < 5n_i, \\
0, & \text{if } c_i \geq 5n_i.
\end{cases}
\]

Equivalently, the same formula can be written in compact max form:

\[
E_i =
\max\left(
0,\,
1 -
\frac{\max(0,\, c_i - 2n_i)}
{3n_i}
\right)
\]

where:

\[
3n_i = 5n_i - 2n_i
\]

The interval from \(2n_i\) to \(5n_i\) is the penalty range.

## 2.5 Penalized Case Evidence Recall

The case evidence recall is multiplied by the API efficiency factor:

\[
\text{PenalizedRecall}_i^{\text{case}}
=
\text{Recall}_i^{\text{case}}
\cdot
E_i
\]

The overall penalized case evidence recall is the average over all test cases:

\[
\text{PenalizedCaseRecall}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\text{Recall}_i^{\text{case}}
\cdot
E_i
\right)
\]

This means:

- **If the system stays within the API budget:**

  \[
  E_i = 1
  \]

  The case recall score is not penalized.

- **If the system makes too many API calls:**

  \[
  E_i < 1
  \]

  The case recall score is reduced.

- **If the system makes at least \(5n_i\) API calls:**

  \[
  E_i = 0
  \]

  The system receives zero recall score for that case.
## 2.6 Micro Law Evidence F1

Law evidence is evaluated using micro-averaged F1 over the full test set.

Micro precision is:

\[
\text{Precision}_{\text{micro}}^{\text{law}}
=
\frac{|P_{\text{law}} \cap G_{\text{law}}|}
{|P_{\text{law}}|}
\]

Micro recall is:

\[
\text{Recall}_{\text{micro}}^{\text{law}}
=
\frac{|P_{\text{law}} \cap G_{\text{law}}|}
{|G_{\text{law}}|}
\]

Micro F1 is:

\[
F1_{\text{micro}}^{\text{law}}
=
\frac{
2 \cdot
\text{Precision}_{\text{micro}}^{\text{law}}
\cdot
\text{Recall}_{\text{micro}}^{\text{law}}
}{
\text{Precision}_{\text{micro}}^{\text{law}}
+
\text{Recall}_{\text{micro}}^{\text{law}}
}
\]

This component rewards systems that retrieve relevant legal provisions from the law corpus.

Precision rewards systems for avoiding irrelevant legal provisions.

Recall rewards systems for finding all required legal provisions.

F1 balances both precision and recall.

## 2.7 Full Final Score Formula

Putting all components together:

\[
\text{FinalScore}
=
0.70 \cdot
\frac{1}{N}
\sum_{i=1}^{N}
\text{Outcome}_i
+
0.20 \cdot
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\text{Recall}_i^{\text{case}}
\cdot
E_i
\right)
+
0.10 \cdot
F1_{\text{micro}}^{\text{law}}
\]

where:

\[
E_i =
\begin{cases}
1, & \text{if } c_i \leq 2n_i, \\
1 - \dfrac{c_i - 2n_i}{3n_i}, & \text{if } 2n_i < c_i < 5n_i, \\
0, & \text{if } c_i \geq 5n_i.
\end{cases}
\]

In compact form:

\[
\text{FinalScore}
=
0.70 \cdot \text{OutcomeAccuracy}
+
0.20 \cdot \text{PenalizedCaseRecall}
+
0.10 \cdot F1_{\text{micro}}^{\text{law}}
\]
## 2.8 Example

Suppose case \(i\) has:

\[
n_i = 20
\]

Then:

\[
2n_i = 40
\]

and:

\[
5n_i = 100
\]

So the API penalty works as follows:

| API calls \(c_i\) | API efficiency \(E_i\) | Explanation |
|------------------:|-----------------------:|-------------|
| 20 | 1.000 | No penalty |
| 40 | 1.000 | No penalty limit |
| 50 | 0.833 | Linear penalty starts |
| 60 | 0.667 | Recall component is reduced |
| 70 | 0.500 | Half of case recall remains |
| 80 | 0.333 | One third of case recall remains |
| 90 | 0.167 | Small portion of case recall remains |
| 100 | 0.000 | Case recall becomes zero |
| 120 | 0.000 | Case recall remains zero |

If:

\[
\text{Recall}_i^{\text{case}} = 0.8
\]

Then:

| API calls \(c_i\) | \(E_i\) | Penalized case recall |
|------------------:|--------:|----------------------:|
| 40 | 1.000 | \(0.8 \cdot 1.000 = 0.800\) |
| 70 | 0.500 | \(0.8 \cdot 0.500 = 0.400\) |
| 100 | 0.000 | \(0.8 \cdot 0.000 = 0.000\) |
