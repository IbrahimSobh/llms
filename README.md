# llms

Purpose of Language Models: Assign a probability to a sentence, to distinguish between the more likely and the less likely sentences.

## Applications:
1. Machine Translation: P(high winds tonight) > P(large winds tonight)
2. Spelling correction: P(about fifteen minutes from) > P(about fifteen minutes from)
3. Speech Recognition: P(I saw a van) > P(eyes awe of an)
4. Summarization, question answering, etc.

For Speech Recognition, we use not only the acoustics model (the speech signal), but also a language model. Similarly, for Optical Character Recognition (OCR), we use both a vision model and a language model. Language models are very important for such recognition systems.

> Sometimes, you hear or read a sentence that is not clear, but using your language model, you still can recognize it at a high accuracy despite the noisy vision/speech input.

The language model computes either of:
- The probability of a sentence or sequence of words: $P(w_1, w_2, w_3, ..., w_n)$
- The probability of an upcoming word: $P(w_5 | w_1, w_2, w_3, w_4)$

The Chain Rule: $P(x_1, x_2, x_3, …, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)…P(x_n|x_1,…,x_{n-1})$



