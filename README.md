# Fine-tuning a small LLM for named entity recognition with QLoRA

This project fine-tunes **Qwen2.5-1.5B-Instruct** to pull named entities out of news text and return them as JSON. It uses QLoRA (a 4-bit quantised base model with small trainable LoRA adapters), so the whole thing runs on a free Google Colab T4 GPU.

> **What is NER?** Named entity recognition (NER) means finding the names of specific things in a piece of text and labelling what kind of thing each one is. For example, in *"Sarah Mensah, a Ghanaian engineer, joined the World Bank in London on 3 March"*, an NER system should find **Sarah Mensah** (Person), **Ghanaian** (Nationality/group), **World Bank** (Organization), **London** (Location) and **3 March** (Date). It's a basic building block for turning unstructured text, such as news, reports and emails, into structured data you can search, filter or analyse.

The goal was to answer a practical question: for a structured extraction task, how far does prompting a small open model get you, and what does fine-tuning add?

**Short answer:** prompting got an overall F1 of **0.18** (zero-shot) and **0.24** (3-shot). After two epochs of LoRA fine-tuning, the same model scored **0.86**.

📓 Notebook: [`ner_lora_qwen.ipynb`](ner_lora_qwen.ipynb)

---

## What we did

### 1. Data

The data is the [Stanford `en-worldwide-newswire`](https://github.com/stanfordnlp/en-worldwide-newswire) NER dataset: 1,075 English-language news articles from outlets around the world, annotated token by token with BIO tags across nine entity types.

| Type | Meaning | Count |
|---|---|---|
| Location | Places | 11,527 |
| Person | People | 9,587 |
| Organization | Companies, agencies, parties, courts | 8,256 |
| NORP | Nationalities, religious or political groups | 4,515 |
| Date | Dates and times | 4,217 |
| Misc | Events, laws, other names | 2,944 |
| Money | Monetary amounts | 952 |
| Facility | Buildings, airports, roads | 567 |
| Product | Products | 260 |

### 2. Turning tags into a generation task

LLMs generate text, so the token-level tags were converted into input/output pairs:

- Sentences were grouped into chunks of up to ~120 tokens.
- Each chunk's tags were collapsed into entity spans and written out as JSON, for example `{"Person": ["David Kato"], "NORP": ["Ugandan"]}`.
- The split was made **by article** (80/10/10): 860 train / 107 validation / 108 test articles, giving 5,444 / 704 / 668 examples. Splitting by article rather than by chunk stops text from the same story appearing in both training and evaluation.

### 3. Baselines

Before training anything, the base model was evaluated with a system prompt that defined the nine types and asked for JSON output:

- **Zero-shot:** the instructions alone.
- **3-shot:** the instructions plus three worked examples from the training set.

### 4. QLoRA fine-tuning

In plain terms: rather than retraining the whole model, we froze it, compressed it to save memory, and trained a small add-on that teaches it this one job. Here's each piece.

**Compressing the model to fit on a free GPU (4-bit NF4 quantisation).**
A model's "knowledge" is stored as about 1.5 billion numbers (its *weights*). Normally each one takes 16 bits of memory. Here they were squashed down to 4 bits each using the `bitsandbytes` library. It's a bit like saving a photo as a heavily compressed JPEG: slightly less precise, but a quarter of the size. NF4 is the particular compression method. This is the "Q" (quantised) in QLoRA.

**Training a small add-on instead of the whole model (LoRA adapters).**
The original 1.5 billion weights are **frozen** and never change. A small set of new weights, called an *adapter*, is bolted on, and only that gets trained. That's about 18 million numbers, roughly 1.2% of the model's size. If you're from a software background, it's like overriding a few methods in a subclass instead of rewriting the base class. The model keeps its general understanding of English, and the adapter learns the specific task: "find the entities and label them the way this dataset does."
- The adapter was attached to every one of the model's main internal layers (the *attention* and *MLP projection* layers), not just some of them.
- *Rank 16* controls how big the adapter is. A bigger adapter can learn more, but it uses more memory.

**The training run.**
Training used Hugging Face's ready-made `SFTTrainer`. *SFT* (supervised fine-tuning) means "here's an input, here's the correct answer, learn to produce it."
- **2 epochs:** the model saw every training example twice.
- **Batch size 16:** it looks at 16 examples, measures how wrong it was on average, then nudges the adapter weights. It repeats that over and over.
- **Learning rate 2e-4 with cosine decay:** the learning rate is how big each nudge is. Cosine decay starts with bigger nudges and gradually shrinks them, like taking big strides towards a target and then small careful steps as you get close.
- **Marked on the answer only:** the model is only scored on the JSON answer it writes, not on whether it can repeat back the article or the instructions.

**A bug fix worth knowing about.** Training initially crashed on the T4 GPU because of a number-precision mismatch in the mixed-precision setup. Keeping the small adapter in full 32-bit precision (fp32) while the rest of the model stayed compressed fixed it. This is a common gotcha when training on T4s.

### 5. Evaluation

All three set-ups were scored on the same 200 validation examples, using the same zero-shot prompt for the fine-tuned model.

- A prediction counts as correct only if both the entity **string** and its **type** match the gold label (case- and whitespace-insensitive).
- Precision, recall and F1 are reported per type and micro-averaged overall.
- If the output isn't valid JSON, every gold entity in that example counts as missed.

---

## Results

| | Valid JSON | Precision | Recall | **F1** |
|---|---|---|---|---|
| Zero-shot | 42.5% | 0.34 | 0.12 | **0.18** |
| 3-shot | 98.5% | 0.31 | 0.19 | **0.24** |
| **Fine-tuned (LoRA)** | **99.5%** | **0.87** | **0.84** | **0.86** |

F1 by entity type:

| Type | Zero-shot | 3-shot | Fine-tuned |
|---|---|---|---|
| Date | 0.12 | 0.14 | **0.95** |
| Money | 0.00 | 0.31 | **0.95** |
| Person | 0.36 | 0.56 | **0.94** |
| Location | 0.27 | 0.22 | **0.89** |
| NORP | 0.00 | 0.01 | **0.89** |
| Organization | 0.06 | 0.02 | **0.75** |
| Misc | 0.04 | 0.03 | **0.50** |
| Facility | 0.00 | 0.29 | **0.31** |
| Product | 0.00 | 0.00 | **0.00** |

---

## What the findings mean

### Few-shot prompting fixed the format, not the task

Going from zero-shot to 3-shot pushed valid JSON from 42.5% to 98.5%, but F1 barely moved (0.18 to 0.24). The examples taught the model *what the output should look like*. They didn't teach it *what counts as which entity*.

The raw outputs make this clear. In one 3-shot prediction the model labelled a political party ("UP"), a nationality ("Equatoguineans") and two cities ("Malabo", "Bata") all as `Person`. It also filled `Misc` with phrases like "protest marches" that aren't named entities at all.

### Fine-tuning taught the annotation scheme

Fine-tuning gave a 3.6× improvement over the best prompt (0.24 to 0.86 F1). It also left precision and recall roughly balanced (0.87 / 0.84), so the model isn't over- or under-predicting.

The biggest gains came where the label definitions are a matter of convention rather than common sense:

- **NORP:** 0.01 to 0.89
- **Organization:** 0.02 to 0.75

A small model can't reliably infer conventions like these from a paragraph of instructions, but it learns them quickly from a few thousand labelled examples.

### What's still weak depends on how much data there is and how clearly a type is defined

| Type | F1 | Why it's hard |
|---|---|---|
| Product | 0.00 | Only 260 examples in the whole dataset, so the model effectively never learned it. |
| Facility | 0.31 | Rare (567 examples) and easily confused with Location or Organization. |
| Misc | 0.50 | A catch-all type with a fuzzy definition, which makes it hard for annotators and models alike. |

Money is a useful contrast. It is also fairly rare (952 examples) but scores 0.95, because amounts follow very regular patterns. Rarity on its own isn't fatal; rarity combined with ambiguity is.

### Why this matters

- **A small model is good enough once it's trained.** A 1.5B-parameter model in 4-bit fits in a few GB of memory. After fine-tuning, it does solid structured extraction on the common entity types. That makes it a realistic option where data can't leave your environment, or where inference cost and latency matter more than raw capability.
- **Fine-tuning is cheap.** The adapter is about 1% of the model's parameters, and training ran on a free Colab GPU. When you have labelled data, fine-tuning is a far more effective use of effort than prompt engineering.
- **Evaluating per type matters.** The headline 0.86 F1 hides a model that is excellent on six types and unreliable on three. In a real system, those weak types would need more data or a fallback before anyone relied on them.

---

## Caveats

- **Small, single-run evaluation.** Results are from 200 validation examples and one training run, with no variance across seeds. The 108-article test split is held out and **has not been used yet**.
- **String matching, not span matching.** Entities are compared as normalised strings per type, as a set within each example. So repeated mentions count once, and the scores aren't directly comparable to standard token-level `seqeval` scores.
- **Hallucinations are not checked separately.** Predicted strings aren't verified to appear in the input text. A hallucinated entity is simply counted as a false positive.

## Next steps

**1. Get a final score from the untouched test set.**
All the scores above come from the *validation* set. That's data I looked at while building and tuning the project, so it's slightly influenced by my choices. The *test* set (108 articles) has been kept aside and never used. Running the final model on it once gives the honest, final number. It's like keeping one exam paper you never practise on.

**2. Compare against a traditional NER model.**
Before LLMs, NER was usually done with smaller models such as BERT. Rather than writing out an answer, these read the text and put a label on each word. They're very good at this exact task and cheap to run. Comparing against one answers the obvious question: was an LLM actually the better tool here, or just the newer one?

**3. Improve the weak entity types.**
Product, Facility and Misc scored badly, mostly because there were so few examples of them. Two ways to fix that:
- **Show them more often:** repeat training examples that contain these types (known as *upsampling*), so the model gets more practice on them.
- **Write clearer definitions:** "Misc" is vague even for a human. Tighter rules in the prompt about what belongs in each type should help.

**4. Catch made-up answers.**
LLMs can invent things. At the moment, if the model "finds" an entity that isn't actually in the article, it's just counted as a wrong answer. A simple check that every predicted name appears word for word in the input would measure how often this happens. In a real system, the same check could filter out made-up answers before anyone sees them.

## Running it

Open the notebook in Google Colab with a T4 GPU runtime and run the cells in order. It clones the dataset, writes the processed splits and checkpoints to Google Drive (`MyDrive/ner-lora/`), runs the baselines, fine-tunes and evaluates.

Main libraries: `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`.
