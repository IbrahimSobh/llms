# Large Language Models (llms)

---

## Introduction: What is a language model?

Simple definition: Language Modeling is the task of predicting what word comes next.

"The dog is playing in the ..."
- park
- woods
- snow
- office
- university
- Neural network
- ? 

The main purpose of **Language Models** is to assign a probability to a sentence, to distinguish between the more likely and the less likely sentences.

### Applications of language models:
1. Machine Translation: P(high winds tonight) > P(large winds tonight)
2. Spelling correction: P(about fifteen minutes from) > P(about fifteen minuets from)
3. Speech Recognition: P(I saw a van) > P(eyes awe of an)
4. Authorship identification: who wrote some sample text
5. Summarization, question answering, dialogue bots, etc.

For Speech Recognition, we use not only the acoustics model (the speech signal), but also a language model. Similarly, for Optical Character Recognition (OCR), we use both a vision model and a language model. Language models are very important for such recognition systems.

> Sometimes, you hear or read a sentence that is not clear, but using your language model, you still can recognize it at a high accuracy despite the noisy vision/speech input.

The language model computes either of:
- The probability of an upcoming word: $P(w_5 | w_1, w_2, w_3, w_4)$
- The probability of a sentence or sequence of words (according to the Language Model): $P(w_1, w_2, w_3, ..., w_n)$


The Chain Rule: $P(x_1, x_2, x_3, …, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)…P(x_n|x_1,…,x_{n-1})$

> $P(The, water, is, so, clear) = P(The) × P(water|The) × P(is|The, water) × P(so|The, water, is) × P(clear | The, water, is, so)$

What just happened? The Chain Rule is applied to compute the joint probability of words in a sentence.

---

## Statistical Language Modeling:

### n-gram Language Models
Using a large amount of text (corpus such as Wikipedia), we collect statistics about how frequently different words are, and use these to predict the next word. For example, the probability that a word _w_ comes after these three words *students opened their* can be estimated as follows: 
- P(w | students opened their) = count(students opened their w) / count(students opened their)

The above example is a 4-gram model. And we may get: 
- P(books | students opened their) = 0.4
- P(cars | students, opened, their) = 0.05
- P(... | students, opened, their) = ...

> We can conclude that the word “books” is more probable than “cars” in this context. 

We ignored the previous context before "students opened their"

Accordingly, arbitrary text can be generated from a language model given starting word(s), by sampling from the output probability distribution of the next word, and so on.

<!--
### How to estimate these probabilities?

Amusing we have a large text corpus (data set like Wikipedia), we can count and divide as follows:

- $P(clear |The, water, is, so) = Count (The, water, is, so, clear) / Count (The, water, is, so)$
-->

Sparsity: Sometimes we do not have enough data to estimate the following: 
- $P(clear |The, water, is, so) = Count (The, water, is, so, clear) / Count (The, water, is, so)$

Markov Assumption (Simplifying assumption):
- $P(clear |The, water, is, so) ≈ P(clear | so)$
- Or $P(clear |The, water, is, so) ≈ P(clear | is, so)$

Formally:
- $P(w_1 w_2 … w_n ) ≈ ∏i P(w_i | w_{i−k} … w_{i−1})$
- $P(w_i | w_1 w_2 … w_{i−1}) ≈ P(w_i | w_{i−k} … w_{i−1})$
- Unigram model: $P(w_1 w_2 … w_n ) ≈ ∏i P(w_i)$
- Bigram model: $P(w_i | w_1 w_2 … w{i−1}) ≈ P(w_i | w_{i−1})$

> We can extend to trigrams, 4-grams, 5-grams, and N-grams.

 In general, this is an insufficient model of language because the language has long-distance dependencies. However, in practice, these 3,4 grams work well for most of the applications.

<!---
### Estimating bigram probabilities:
The Maximum Likelihood Estimate (MLE): of all the times we saw the word wi-1, how many times it was followed by the word wi

$P(w_i | w_{i−1}) = count(w_{i−1}, w_i) / count(w_{i−1})$
Practical Issue: We do everything in log space to avoid underflow
$log(p1 × p2 × p3 × p4 ) = log p1 + log p2 + log p3 + log p4$
-->

### Building Statistical Language Models:

#### Toolkits

- [SRILM](http://www.speech.sri.com/projects/srilm/) is a toolkit for building and applying statistical language models, primarily for use in speech recognition, statistical tagging and segmentation, and machine translation. It has been under development in the SRI Speech Technology and Research Laboratory since 1995.
- [KenLM](https://kheafield.com/code/kenlm/) is a fast and scalable toolkit that builds and queries language models.

#### N-gram Models

Google's N-gram Models Belong to You: Google Research has been using word n-gram models for a variety of R&D projects. [Google N-Gram](https://ai.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html) processed 1,024,908,267,229 words of running text and published the counts for all 1,176,470,663 five-word sequences that appear at least 40 times.

The counts of text from the Linguistics Data Consortium [LDC](https://www.ldc.upenn.edu/) are as follows:

```
File sizes: approx. 24 GB compressed (gzip'ed) text files

Number of tokens:    1,024,908,267,229
Number of sentences:    95,119,665,584
Number of unigrams:         13,588,391
Number of bigrams:         314,843,401
Number of trigrams:        977,069,902
Number of fourgrams:     1,313,818,354
Number of fivegrams:     1,176,470,663
```


The following is an example of the **4-gram** data in this corpus:

```
serve as the incoming 92
serve as the incubator 99
serve as the independent 794
serve as the index 223
serve as the indication 72
serve as the indicator 120
serve as the indicators 45
serve as the indispensable 111
serve as the indispensible 40
```

For example, the sequence of the four words "serve as the indication" has been seen in the corpus 72 times.

<!---
Try some examples of your own using [Google Books Ngram Viewer](https://books.google.com/ngrams/) and see the frequency of likely and unlikely N-grams.

![ngramviewer.png](images/ngramviewer.png)
-->

---

## Evaluation: How good is our model?

> Does our language model prefer good (likely) sentences to bad ones?

### Extrinsic evaluation:

1. For comparing models A and B, put each model in a task (spelling, corrector, speech recognizer, machine translation)
2. Run the task and compare the accuracy for A and for B
3. Best evaluation but not practical and time consuming!

### Intrinsic evaluation:

- **Intuition**: The best language model is one that best predicts an unseen test set (assigns high probability to sentences).
- **Perplexity** is the standard evaluation metric for Language Models.
- **Perplexity** is defined as the inverse probability of a text, according to the Language Model.
- A good language model should give a lower Perplexity for a test text. Specifically, a lower perplexity for a given text means that text has a high probability in the eyes of that Language Model.

> Perplexity is the inverse probability of the test set, normalized by the number of words


![preplexity02.png](images/preplexity02.png)

> Lower perplexity = Better model

> Perplexity is related to branch factor: On average, how many things could occur next.


### Limitations of Statistical Language models
 
- What if “students opened their” never occurred in data? (Sparsity problem) We may condition on “opened their” instead (_backoff_).
- What if “students opened their ” never occurred in data? We may add a small 𝛿 to the count for every w (_smoothing_).
- Large storage requirements: Need to store count for all n-grams you saw in the corpus.

---

## Neural Language Models (NLM)

- **What is the problem?** One of the biggest challenges in natural language processing (NLP) is the shortage of training data for many distinct tasks. However, modern deep learning-based NLP models improve when trained on millions, or billions, of annotated training examples.

- **Pre-training is the solution:** To help close this gap, a variety of techniques have been developed for training general-purpose language representation models using the enormous amount of unannotated text. The pre-trained model can then be fine-tuned on small data for different tasks like question answering and sentiment analysis, resulting in substantial accuracy improvements compared to training on these datasets from scratch.

NLM usually (but not always) uses an RNN to learn sequences of words (sentences, paragraphs, … etc) and hence can predict the next word. 

Advantages: 
- Can process variable-length input
- Computations for step t use information from many steps back
- Model size doesn’t increase for longer input, the same weights are applied on every timestep.

![nlm01.png](images/nlm01.png)

As depicted, At each step, we have a probability distribution of the next word over the vocabulary.
Disadvantages: 
- Recurrent computation is _slow_ (sequential, one step at a time)
- In practice, for long sequences, difficult_ to access information_ from many steps back

### Transformer-based Language models   

The Transformer architecture was proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), used for the Neural Machine Translation task (NMT), consisting of: 
- **Encoder**: Network that encodes the input sequence.
- **Decoder**: Network that generates the output sequences conditioned on the input.

As mentioned in the paper: 
> "_We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely_"

The main idea of **attention** can be summarized as mentioned in the OpenAi's [article](https://openai.com/blog/sparse-transformer/):
> "_... every output element is connected to every input element, and the weightings between them are **dynamically calculated based upon the circumstances**, a process called attention._"

Based on this architecture (the vanilla Transformers!), **encoder or decoder** components can be used alone to enable massive pre-trained generic models that can be fine-tuned for downstream tasks such as text classification, translation, summarization, question answering, etc. For Example:

- "Pre-training of Deep Bidirectional Transformers for Language Understanding" [BERT](https://arxiv.org/abs/1810.04805) is mainly based on the encoder architecture trained on massive text datasets to predict randomly masked words and "is-next sentence" classification tasks.
- [GPT](https://arxiv.org/pdf/2005.14165.pdf), on the other hand, is an auto-regressive generative model that is mainly based on the decoder architecture, trained on massive text datasets to predict the next word (unlike BERT, GPT can generate sequences).

> These models, BERT and GPT for instance, can be considered as the NLP's ImageNET.

![bertvsgpt.png](images/bertvsgpt.png)

As shown, BERT is deeply bidirectional, OpenAI GPT is unidirectional, and ELMo is shallowly bidirectional.

Pre-trained representations can be:
- **Context-free**: such as word2vec or GloVe that generates a single/fixed word embedding (vector) representation for each word in the vocabulary (independent of the context of that word at test time)
- **Contextual**: generates a representation of each word based on the other words in the sentence.

Contextual Language models can be:
- **Causal language model (CML)**: Predict the next token passed on previous ones. (GPT)
- **Masked language model (MLM)**: Predict the masked token based on the surrounding contextual tokens (BERT)
  
<!--
#### To do
- Code Bert https://colab.research.google.com/drive/17sJR6JwoQ7Trr5WsUUIpHLZBElf8WrVq?usp=sharing#scrollTo=-u2Feyk5Gg7o
- Code GPT
- Code Falcon
- Code GPT4ALL
- Code CodeTF
- Chat with my docs
- etc.
-->

---

## 💥 Practical LLMs

In this part, we are going to use different large language models 

### 🚀 Hello GPT2 

<a target="_blank" href="https://colab.research.google.com/drive/1eBcoHjJ2S4G_64sBvYS8G8B-1WSRLQAF?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (a successor to GPT) is a pre-trained model on English language using a causal language modeling (**CLM**) objective, trained simply to predict the next word in 40GB of Internet text. It was first released on this [page](https://openai.com/research/better-language-models). GPT2 displays a broad set of capabilities, including the ability to generate conditional synthetic text samples. On language tasks like question answering, reading comprehension, summarization, and translation, GPT2 _begins_ to learn these tasks from the raw text, using no task-specific training data. DistilGPT2 is a distilled version of GPT2, it is intended to be used for similar use cases with the increased functionality of being smaller and easier to run than the base model.

Here we load a pre-trained **GPT2** model, ask the GPT2 model to continue our input text (prompt), and finally, extract embedded features from the DistilGPT2 model. 

```
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
generator("The capital of Japan is Tokyo, The capital of Egypt is", max_length=13, num_return_sequences=2)
```

```
[{'generated_text': 'The capital of Japan is Tokyo, The capital of Egypt is Cairo'},
{'generated_text': 'The capital of Japan is Tokyo, The capital of Egypt is Alexandria'}]
```

### 🚀 Hello BERT 

<a target="_blank" href="https://colab.research.google.com/drive/1n8fd41Bi8yaWp0__eVCe9T4ctWVwSxvC?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[BERT](https://arxiv.org/abs/1810.04805) is a transformers model pre-trained on a large corpus of English data in a self-supervised fashion. This means it was pre-trained on the raw texts only, with no humans labeling them in any way with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:
1. Masked language modeling (**MLM**): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally masks the future tokens. It allows the model to learn a bidirectional representation of the sentence.
2. Next sentence prediction (**NSP**): the model concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.


In this example, we are going to use a pre-trained **BERT** model for the sentiment analysis task.

1. Baseline bidirectional LSTM model (accuracy = 65%)
2. Use BERT as a feature extractor using only [CLS] feature (accuracy = 81%)
3. Use BERT as a feature extractor for the sequence representation (accuracy = 85%)

```
import transformers as ppb

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)
```

### 🚀 GPT4ALL

<a target="_blank" href="https://colab.research.google.com/drive/1AONl_3F8C6fB2NKaQsuAHio0sdjulIRk?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[GPT4All](https://docs.gpt4all.io/) is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs.

```
import gpt4all
gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy.bin")

with gptj.chat_session():
    response = gptj.generate(prompt='hello', top_k=1)
    response = gptj.generate(prompt='My name is Ibrahim, what is your name?', top_k=1)
    response = gptj.generate(prompt='What is the capital of Egypt?', top_k=1)
    response = gptj.generate(prompt='What is my name?', top_k=1)
    print(gptj.current_chat_session) 
```

```
[{'role': 'user', 'content': 'hello'}, 
{'role': 'assistant', 'content': 'Hello! How can I assist you today?'}, 

{'role': 'user', 'content': 'My name is Ibrahim, what is your name?'}, 
{'role': 'assistant', 'content': 'I am an artificial intelligence assistant. My name is AI-Assistant.'}, 

{'role': 'user', 'content': 'What is the capital of Egypt?'}, 
{'role': 'assistant', 'content': 'The capital city of Egypt is Cairo.'}, 

{'role': 'user', 'content': 'What is my name?'}, 
{'role': 'assistant', 'content': 'Your name is Ibrahim, what a beautiful name!'}]
```

### 🚀 Falcon

<a target="_blank" href="https://colab.research.google.com/drive/1bkBWa38-kO9T-8mvI1iXAQCIdOO19Uv2?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[Falcon](https://huggingface.co/tiiuae) LLM is TII's flagship series of large language models, built from scratch using a custom data pipeline and distributed training. Falcon-7B/40B models are state-of-the-art for their size, outperforming most other models on NLP benchmarks. Open-sourced a number of artefacts:
- The Falcon-7/40B pretrained and instruct models, under the Apache 2.0 software license. 

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

```
Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
Daniel: Hello, Girafatron!
Girafatron: Hi Daniel! I am Girafatron, the world's first Giraffe. How can I be of assistance to you, human boy?
Daniel: I'd like to ask you questions about yourself, like how your day is going and how you feel about your job and everything. Would you like to talk about that?
Girafatron: Sure, my day is going great. I'm feeling fantastic. As for my job, I'm enjoying it!
Daniel: What do you like most about your job?
Girafatron: I love being the tallest animal in the universe! It's really fulfilling.
```

### 🚀 CodeT5+

<a target="_blank" href="https://colab.research.google.com/drive/1Ik8w6BgHazuf45E5GrZd0vyx6SV3EOzG?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) is a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (i.e. encoder-only, decoder-only, and encoder-decoder) to support a wide range of code understanding and generation tasks.

```
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "Salesforce/codet5p-770m-py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def factorial(n):", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

```
def factorial(n):
'''
Returns the factorial of a given number.
'''
if n == 0:
    return 1
return n * factorial(n - 1)

def main():
    '''
    Tests the factorial function.
    '''
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 120
    assert factorial(5) == 720
    assert factorial(6) == 5040
    assert factorial(7) == 5040
```

For more models, check [CodeTF](https://github.com/salesforce/CodeTF) from Salesforce, a Python transformer-based library for code large language models (Code LLMs) and code intelligence, providing a seamless interface for training and inferencing on code intelligence tasks like code summarization, translation, code generation, and so on.

---

## 💥 More LLMs

🏔️ [Chat with Open Large Language Models](https://chat.lmsys.org/) 

- **Vicuna**: a chat assistant fine-tuned from LLaMA on user-shared conversations by LMSYS	WizardLM: an instruction-following LLM using evol-instruct by Microsoft
- **Guanaco**: a model fine-tuned with QLoRA by UW
- **MPT-Chat**: a chatbot fine-tuned from MPT-7B by MosaicML
- **Koala**: a dialogue model for academic research by BAIR
- **RWKV-4-Raven**: an RNN with transformer-level LLM performance
- **Alpaca**: a model fine-tuned from LLaMA on instruction-following demonstrations by Stanford
- **ChatGLM**: an open bilingual dialogue language model by Tsinghua University
- **OpenAssistant** (oasst): an Open Assistant for everyone by LAION
- **LLaMA**: open and efficient foundation language models by Meta
- **Dolly**: an instruction-tuned open large language model by Databricks
- **FastChat-T5**: a chat assistant fine-tuned from FLAN-T5 by LMSYS	


--- 

## Chat with your documents 
to do 

---

## Further readings:

- [Book: Speech and Language Processing; Daniel Jurafsky](https://www.amazon.com/Speech-Language-Processing-Daniel-Jurafsky/dp/0131873210)
- [Video: Natural Language Processing](https://youtu.be/iWea12EAu6U?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
- [Gentle Introduction to Statistical Language Modeling and Neural Language Models](https://machinelearningmastery.com/statistical-language-modeling-and-neural-language-models/)

