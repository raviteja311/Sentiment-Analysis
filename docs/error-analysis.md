# Error analysis: what the served transformer gets wrong

**Status: sample drawn, tagging not done yet.** The counts in "What the tags
say" below are filled in by hand, from `python -m src.errors summarise`.

## The sample

`python -m src.errors export` drew 100 of the served model's 3,551 test errors
(seed 42, uniform over errors, so the mix matches the model's) into
`reports/errors.csv`. Measured before any tagging:

| Gold -> predicted | Errors in sample |
|---|---|
| neutral -> negative | 39 |
| positive -> neutral | 24 |
| negative -> neutral | 22 |
| neutral -> positive | 15 |

Every sampled error involves the neutral class; not one confuses negative with
positive. The model rarely gets the polarity backwards. What it gets wrong is
**whether a tweet carries sentiment at all**, and it errs towards negative when
it does: neutral read as negative is the largest single group.

`reports/errors.csv` is not committed while it is untagged, because it can be
regenerated exactly. `python -m src.errors export` reads the saved test
predictions in `reports/predictions/` and the TweetEval test split, and with the
fixed seed it writes the same 100 rows as the table above. `export` overwrites
the file, tags included, so commit `reports/errors.csv` as soon as tagging
starts and do not run `export` again after that.

## How to tag

Read each row of `reports/errors.csv` and write one tag in the `tag` column,
and anything worth remembering in `notes`. Tag what makes the tweet hard, not
what the model did. `summarise` refuses any tag outside this list:

| Tag | Use when |
|---|---|
| `sarcasm` | it says the opposite of what it means |
| `negation` | a negator was ignored or misread |
| `mixed` | positive and negative content in the same tweet |
| `neutral-boundary` | mild sentiment or news-like text; neutral vs not is arguable |
| `label-noise` | the gold label looks wrong to a careful reader |
| `emoji` | the sentiment is carried by emoji or emoticons |
| `context` | it needs world knowledge or the thread it replies to |
| `other` | none of the above; say what in `notes` |

Decide `label-noise` before anything else: if the gold label is wrong, the
model may be right.

## What the tags say

_To be filled in from `python -m src.errors summarise`._

## Against the behavioural checks

The served transformer passes all five behavioural checks in `src/behaviour.py`
(18 cases: negation flips, entities, emoji, mention and URL invariance). So any
frequent `negation` or `emoji` tag above is a failure the synthetic cases do not
reproduce, and is worth turning into new cases taken from the real tweets.
