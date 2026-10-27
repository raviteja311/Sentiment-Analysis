# Phase 2: the fine-tune grid and the serving rule, fixed before any run

Written and committed before a single grid run was trained, so the git history
shows the rule could not have been fitted to the results. Changing anything
below after results exist means saying so, and why, in the model card.

## The question

What does fine-tuning add, and why did the first fine-tune lose to the
checkpoint it started from? Two explanations were never separated: the
`glove-v1` preprocessing, unlike what the checkpoint was trained on, and
training a second time on data the checkpoint had already been fitted to.

## The grid

One thing changes between neighbouring runs. Every run is trained with three
seeds: 13, 42 and 1234.

| Run | Start checkpoint | Preprocessing | What it tests |
|---|---|---|---|
| A' | `cardiffnlp/twitter-roberta-base-sentiment` (sentiment-tuned) | `glove-v1` | the served recipe, re-run under this grid's settings |
| B | same | `cardiff-v1` | A' vs B: was the preprocessing mismatch the cause? |
| C | `cardiffnlp/twitter-roberta-base` (Twitter-pretrained, no sentiment labels) | `cardiff-v1` | what fine-tuning adds from a model that never saw the labels |
| D | `roberta-base` (generic English) | `cardiff-v1` | C vs D: what the Twitter pretraining is worth |

A' replaces the original single-seed run as the comparison for B, so that the
two differ in preprocessing alone and not also in seed, warm-up, batch size or
selection split.

## Fixed hyperparameters

Learning rate 2e-5, batch size 32, up to 3 epochs, max length 128, linear
warm-up over the first 6% of steps, weight decay 0.01, fp16 on GPU. Each run
keeps the epoch with the best **macro recall on the dev split** (5% of train,
stratified, seed 42; `src.data.with_dev_split`), with early stopping after one
epoch without improvement. Test is never read during training.

Caveat, stated in advance: the dev tweets come from TweetEval train, which the
sentiment-tuned checkpoint was trained on. For A' and B their dev scores are
therefore optimistic, and dev cannot fairly compare them with C and D. It is
used only to choose an epoch within a run. The comparison between runs is the
single read of test below.

## The serving rule

Candidates: the base checkpoint as published, scored zero-shot, and runs A',
B, C and D.

1. For each run, average test macro recall (TweetEval's official metric) over
   its three seeds.
2. The winner is the run with the highest mean.
3. Compute a paired bootstrap 95% interval for the winner's mean macro recall
   minus the base checkpoint's, resampling test tweets (1,000 resamples, seed
   42) and averaging the winner's three seeds inside each resample.
4. If the interval's lower bound is above zero, serve the winner's best-dev
   seed. Otherwise serve the base checkpoint: equal or better, and simpler,
   since it needs no training run of ours.

`cardiffnlp/twitter-roberta-base-sentiment-latest` is not a candidate. Its
test score was read in Phase 1, before this rule was written, so letting it
compete would be choosing with test knowledge in hand.

## What is reported regardless of the rule

Mean and standard deviation over seeds of test macro recall, macro F1 and
accuracy for every run, the seed-averaged bootstrap interval, the fitted
temperature and raw test ECE (does a well-run fine-tune stay calibrated?), and
the per-epoch dev curves.
