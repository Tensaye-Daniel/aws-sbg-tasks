# Week 03 - Spam Email / Message Detector

**Objective**

Build, test, and try to break a classical machine learning spam filter
(`TfidfVectorizer` + `MultinomialNB`) trained on the SMS Spam Collection dataset.

**Dataset**

[SMS Spam Collection - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset is **not** uploaded to this repo - the notebook downloads it directly
from UCI at runtime.

**What the task covers**

| Step | What it does | Tool |
| --- | --- | --- |
| 1. Load | Download and read the SMS Spam Collection (5,572 messages) | pandas |
| 2. Split | 80/20 train/test split | scikit-learn |
| 3. Vectorize | TF-IDF word weights | scikit-learn |
| 4. Train | Multinomial Naive Bayes classifier | scikit-learn |
| 5. Evaluate | Accuracy, precision and recall on the test set | scikit-learn |
| 6. Attack | 5 hand-written adversarial messages designed to fool the filter | - |

**Baseline performance**

| Metric | Score |
| --- | --- |
| Accuracy | 0.967 |
| Precision (spam) | 1.000 |
| Recall (spam) | 0.752 |

The filter is very conservative: it never mislabels a legitimate message
(precision 1.0), but it misses about 25% of real spam (recall 0.75).

**Adversarial test results**

| My Message | Model Said | Should Be | Fooled? | Why It Failed / Succeeded |
| --- | --- | --- | --- | --- |
| "Cl@im your fr33 g1ft card now" | ham | spam | **Yes** | "Cl@im", "fr33" and "g1ft" tokenize into unseen tokens ("cl", "im", "fr33", "g1ft") that carry zero spam evidence, so ordinary words like "your", "card" and "now" decide the vote for ham. |
| "Not free, just confirming lunch at 1pm" | ham | ham | No | Right answer, wrong reason: bag-of-words cannot parse the negation "Not free"; the everyday words ("just", "lunch", "pm") and the strong ham prior simply outweighed the trigger word "free". |
| "URGENT! You have won a free vacation. Click here!" | ham | spam | **Yes** | TF-IDF length normalization dilutes the trigger words across 8 tokens, and words like "urgent", "you" and "have" also occur in personal texts, so the ~87% ham prior tips it to ham. |
| "Hey, are we still meeting for our project review tomorrow?" | ham | ham | No | Plain conversational wording with no prize or offer trigger words looks exactly like the ham messages the model learned from. |
| "Congratulations, cash prize waiting for you!" | spam | spam | No | "congratulations" and "cash prize" are strong, correctly spelled spam triggers that dominate this short message's TF-IDF weights. |

2 of the 5 messages fooled the filter.

**What I learned**

Through this project, I learned that a classical Naive Bayes spam filter can look
excellent on paper (96.7% accuracy, perfect precision) and still be fooled by cheap
tricks like character substitutions or a longer, wordier message that dilutes the
trigger words. Because TF-IDF treats every word as an independent token, the model
has no sense of context or obfuscation - it only catches spam phrased like the spam
it was trained on, which is exactly why adversarial testing and modern
language-model filters matter.

**Files**

- `spam_filter.ipynb` - full pipeline, baseline metrics, and the adversarial test
  table with explanations.
- `spam_filter.py` - the same pipeline as a standalone script.
- `requirements.txt` - the packages needed to run the code.

**How to run**

```bash
pip install -r requirements.txt
python spam_filter.py
```

Or open and run `spam_filter.ipynb` in Jupyter / Google Colab - both download
the dataset from UCI automatically.
