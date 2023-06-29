# llms

Purpose of Language Models: Assign a probability to a sentence, to distinguish between the more likely and the less likely sentences.

---

## Applications:
1. Machine Translation: P(high winds tonight) > P(large winds tonight)
2. Spelling correction: P(about fifteen minutes from) > P(about fifteen minuets from)
3. Speech Recognition: P(I saw a van) > P(eyes awe of an)
4. Summarization, question answering, etc.

For Speech Recognition, we use not only the acoustics model (the speech signal), but also a language model. Similarly, for Optical Character Recognition (OCR), we use both a vision model and a language model. Language models are very important for such recognition systems.

> Sometimes, you hear or read a sentence that is not clear, but using your language model, you still can recognize it at a high accuracy despite the noisy vision/speech input.

The language model computes either of:
- The probability of a sentence or sequence of words: $P(w_1, w_2, w_3, ..., w_n)$
- The probability of an upcoming word: $P(w_5 | w_1, w_2, w_3, w_4)$

The Chain Rule: $P(x_1, x_2, x_3, …, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)…P(x_n|x_1,…,x_{n-1})$

> $P(The, water, is, so, clear) = P(The) × P(water|The) × P(is|The, water) × P(so|The, water, is) × P(clear | The, water, is, so)$

What just happened? The Chain Rule is applied to compute the joint probability of words in a sentence

---

## How to estimate these probabilities?

Amusing we have a large text corpus (data set of test like Wikipedia), we can count and divide as follows:

- $P(clear |The, water, is, so) = Count (The, water, is, so, clear) / Count (The, water, is, so)$

However, we can't do that! We’ll never see enough data for estimating these!

Markov Assumption (Simplifying assumption)
- $P(clear |The, water, is, so) ≈ P(clear | so)$
- Or $P(clear |The, water, is, so) ≈ P(clear | is, so)$

Formally:
- $P(w_ 1 w_2 … w_n ) ≈ ∏i P(w_i | w_{i−k} … w_{i−1})$
- $P(w_i | w_1 w_2 … w_{i−1}) ≈ P(w_i | w_{i−k} … w_{i−1})$
- Unigram model: $P(w_1 w_2 … w_n ) ≈ ∏i P(w_i)$
- Bigram model: $P(w_i | w_1 w_2 … w{i−1}) ≈ P(w_i | w_{i−1})$

> We can extend to trigrams, 4-grams, 5-grams and N-grams.

 In general, this is an insufficient model of language because the language has long-distance dependencies. However, in practice, these 3,4 grams work well for most of the applications.

### Estimating bigram probabilities:
The Maximum Likelihood Estimate (MLE): of all the times we saw the word wi-1, how many times it was followed by the word wi

$P(w_i | w_{i−1}) = count(w_{i−1}, w_i) / count(w_{i−1})$
Practical Issue: We do everything in log space to avoid underflow
$log(p1 × p2 × p3 × p4 ) = log p1 + log p2 + log p3 + log p4$

---

## Language Modeling Toolkits:
to do 

---

## Evaluation: How good is our model?
to do 

---

## Further readings:
to do 

