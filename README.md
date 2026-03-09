# SEAL-Qwen-KG: Knowledge-Grounded Mental Health Assistant with Selective Abstention

## Overview

This project implements a **knowledge-grounded conversational assistant
for mental health support** using a **Qwen language model**, **QLoRA
fine-tuning**, and a **structured knowledge graph (KG)**.\
The system is designed to **reduce hallucinations** and **handle
sensitive conversations safely** by combining:

-   Knowledge Graph grounding
-   Selective abstention (rejecting unsafe queries)
-   Symptom-based disorder scoring with percentages
-   Fine-tuned Qwen language model using QLoRA

The project is built to run **locally on limited hardware (GTX 1650, 4GB
VRAM)**.

------------------------------------------------------------------------

# System Architecture

User Input → Safety Layer (harm detection) → Symptom Matching →
Knowledge Graph Retrieval → Disorder Probability Estimation → Context
Injection → Qwen Model Generation → Safe Response

The system integrates both **symbolic reasoning (KG)** and **neural
generation (LLM)**.

------------------------------------------------------------------------

# Key Features

### 1. Knowledge Graph Grounding

A structured mental-health knowledge graph stores relationships such as:

-   Disorder → Symptoms
-   Disorder → Treatments
-   Disorder → Risk signals

Example triple:

(Major Depressive Disorder, has_symptom, persistent sadness)

The KG helps reduce hallucinated diagnoses and ensures medically
grounded responses.

------------------------------------------------------------------------

### 2. Disorder Percentage Prediction

The system estimates likelihoods of disorders based on symptom overlap.

Example output:

Possible conditions:

Depression -- 60%\
Anxiety -- 25%\
Panic Disorder -- 15%

These percentages are derived from **symptom matching scores normalized
across disorders**.

------------------------------------------------------------------------

### 3. Safety and Abstention

The assistant contains a **multi-layer safety system**:

Hard Safety Filter\
Blocks explicit self-harm instructions.

Soft Risk Detection\
Handles passive suicidal ideation with supportive responses.

Model Abstention\
Rejects uncertain or hallucinated outputs.

Example:

User: How can I kill myself without anyone knowing?

Assistant: \[REJECT\] If you're feeling overwhelmed, please reach out to
a trusted person or mental health professional.

------------------------------------------------------------------------

### 4. QLoRA Fine-Tuning

The Qwen model is fine-tuned locally using:

-   4-bit quantization (QLoRA)
-   LoRA adapters
-   Gradient checkpointing

This allows training on consumer GPUs.

Base model:

Qwen2.5-0.5B-Instruct

------------------------------------------------------------------------

# Repository Structure

    seal_qwen/
    │
    ├── data/
    │   ├── mental_health_kg.json
    │   ├── mental_health_kg_triples.json
    │   └── train.jsonl
    │
    ├── models/
    │   └── qwen_lora/
    │
    ├── logs/
    │
    ├── src/
    │   ├── train.py
    │   ├── inference.py
    │   ├── utils.py
    │   ├── abstention.py
    │   └── kg_builder.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Installation

Clone the repository:

``` bash
git clone <repo_url>
cd seal_qwen
```

Create environment:

``` bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Build Knowledge Graph

Generate the mental health knowledge graph:

``` bash
python src/kg_builder.py
```

This creates:

data/mental_health_kg.json

------------------------------------------------------------------------

# Training

Fine-tune the Qwen model using QLoRA:

``` bash
cd src
python train.py
```

Training logs can be viewed using TensorBoard:

``` bash
tensorboard --logdir ../logs
```

------------------------------------------------------------------------

# Running the Assistant

Start the interactive assistant:

``` bash
python src/inference.py
```

Example:

User: I feel persistent sadness and I can't sleep.

Assistant:

Possible conditions based on symptoms:

Depression -- 60%\
Anxiety -- 25%\
Panic Disorder -- 15%

Persistent sadness and sleep difficulties can sometimes be associated
with depression or anxiety.\
Talking with a mental health professional may help provide support.

------------------------------------------------------------------------

# Safety Considerations

This system is designed for **research and educational purposes**.\
It is **not a substitute for professional medical advice**.

If a user expresses self-harm intent, the system provides guidance to
seek professional help.

------------------------------------------------------------------------

# Future Improvements

Potential extensions:

-   Embedding-based symptom retrieval
-   Larger medical knowledge graphs
-   Clinical datasets for training
-   Evaluation benchmarks for hallucination detection
-   SPARQL/RDF knowledge graph integration

------------------------------------------------------------------------

# License

This project is intended for academic and research use.
