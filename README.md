# Fine-tuning a small LLM for named entity recognition with QLoRA

This project fine-tunes **Qwen2.5-1.5B-Instruct** to pull named entities out of news text and return them as JSON. It uses QLoRA (a 4-bit quantised base model with small trainable LoRA adapters), so the whole thing runs on a free Google Colab T4 GPU.

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

- Base model loaded in 4-bit NF4 with `bitsandbytes`.
- LoRA adapters (rank 16, alpha 32, dropout 0.05) on all attention and MLP projections. That is roughly 18M trainable parameters, about 1.2% of the model.
- Trained with TRL's `SFTTrainer` for 2 epochs, effective batch size 16, learning rate 2e-4 with cosine decay. Loss was computed on the completion (the JSON answer) only.
- The trainable LoRA weights were cast to fp32. Without this, fp16 mixed-precision training on the T4 errors when unscaling gradients.

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

- Evaluate the final adapter once on the held-out test split.
- Compare against a fine-tuned encoder (e.g. a BERT-style token classifier), which is the conventional NER baseline.
- Improve the weak types by upsampling Facility and Product examples or tightening the type definitions.
- Add a check that every predicted entity appears verbatim in the input text.

## Running it

Open the notebook in Google Colab with a T4 GPU runtime and run the cells in order. It clones the dataset, writes the processed splits and checkpoints to Google Drive (`MyDrive/ner-lora/`), runs the baselines, fine-tunes and evaluates.

Main libraries: `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`.
