---
layout: post
title: "Do LLMs know what Medical terms mean?"
date: 2025-05-12 10:00:00 +0000
categories: [LLM, AI, Explaniable AI, Factual Understanding]
tags: [LLM, AI, Explaniable AI, Factual Understanding]
math: true  # Enable math equations rendering
author: Pramodith B
by: Pramodith B
---

# Do LLMs know the meaning of Medical Terms?

There's been a never-ending debate about whether LLMs understand the data that they process, so much so that people have started to debate what _understanding_ something actually means. While some argue that LLMs are mere statistical machines that capture patterns in when and where different words are used, others claim that LLMs do indeed have a sense for the underlying meanings of words.

One thing that's undeniable is that LLMs need to remember/store facts to perform well on most tasks. The definitions of
words in my humble opinion are facts. Knowing the definitions of words lays down the foundation for an intelligence to be able to use them correctly in context and perhaps even understand them.

Medicine is a domain where it's critical to know what different terms mean. If a doctor doesn't know the difference between a stroke and a heart attack, you'd be very skeptical about their ability to recognize and diagnose patients with suffering from one or the other. 

In this blog we'll evaluate different LLMs on their ability to recognize the definitions of medical terms.

## Task Description
Our task is straight forward given a definition of a medical term, we ask the LLM to predict the word(s) that best describe the definition.

The prompt used is:

```
You are a word prediction model.You will be given a definition and the part of speech of the word (if available).You must predict the word being defined.

Respond with just the word and no additional text.# Examples:
Definition: A custom-made or tailored item.
Part of speech: adjective
Answer: Bespoke
Definition: To rage in excess of.
Part of speech: verb
Answer: Outrage
Definition: {definition}
Part of speech: {part_of_speech}
Answer:
```

## Dataset Creation
We leverage the Merriam Webster [Medical Dictionary](https://dictionaryapi.com/products/api-medical-dictionary) API to fetch definitions of medical terms.

The api requests require two arguments:
1. 'word': The word whose definition is to be fetched
2. 'api_key': The api key to authenticate the request

In order to get a list of medical terms we use an [open source repo](https://github.com/glutanimate/wordlist-medicalterms-en/blob/master/wordlist.txt) that's scraped medical terms from:

* OpenMedSpel by e-MedTools
* Raj&Co-Med-Spel-Chek by Rajasekharan N. of Raj&Co

The dataset is then processed to get a csv file with the following columns:

* word: The medical term
* definition: The definition of the medical term
* part_of_speech: The part of speech of the medical term
* synonyms: The synonyms of the medical term

The processing and interpretation of the fields of the API are a bit convoluted and reader's interested in learning more can refer to the [code](https://github.com/pramodith/definition_understanding/blob/pramodith/get_word_definitions/src/data_processing/collect_dictionary.py).

It's also important to note that in the context of this dataset the **synonyms actually map to words/terms that share the same stem** of the term rather than true synonyms. An example of synonyms that share the same stem is
```
cataplasia -> 'cataplasias', 'cataplastic'
```

We also take additional steps to make sure that the neither the term or any of its stems occur in the definition and have length filters of >=50 and <=1000 characters on the definitions.

The API has a soft limit of 1000 requests per day, we build a dataset of 1100 examples. The dataset can be found [here](https://github.com/pramodith/definition_understanding/blob/pramodith/get_word_definitions/data/processed_dictionary.csv).

## Model Evaluation
We evaluate the models using the following metrics:

* Exact Match: Case insensitvie exact match between the term and the prediction.
* Fuzzy Match: A simple bi-directional `is in` check on the prediction and the medical term.
* Synonym Match: Check if the predicted term matches any of the stems.
* JudgeLLM Match: We use a JudgeLLM (GPT-4.1) to evaluate if the prediction and the medical term are synonymous given the definition. This helps us account for synonyms that don't share the same stem.

It's also important to note that while computing fuzzy, synonym and fuzzy match are all inclusive of an exact match.
```
fuzzy_accuracy = exact_accuracy + fuzzy_match_score
synonym_accuracy = exact_accuracy + synonym_match_score
judgellm_accuracy = exact_accuracy + judge_llm_match_score
```

### Model Providers
The current code-base supports querying OpenAI, Anthropic, Gemini and models hosted on Together.ai using litellm.
We leverage `vLLM` to run local open source models. All local models were run on an A40 GPU which has 44GB of vRAM.


### Generation Parameters
In order to simulate greedy decoding we set the temperature to 0 and top_p to 0.001. Such a small top_p almost guarantees that just the token with the highest logit is selected at each decoding step.


### JudgeLLM Prompt
The prompt used for the JudgeLLM is:
```
You are an expert linguist. You will judge if two words are the same or synonymous given a definition. Respond with a single word: 'Yes' or 'No'.

## Examples
    Definition: Bleeding from the nose, usually due to ruptured blood vessels in the nasal mucosa.
    Are the words 'Epistaxis' and 'Nosebleed' the same or synonymous? Yes
    Definition: The largest part of the brain, responsible for higher brain functions like thought, action, and sensory processing.
    Are the words 'Cerebrum' and 'Forebrain' the same or synonymous? Yes
    Definition: An elevated body temperature, often due to infection or illness.
    Are the words 'Fever' and 'diarrhea' the same or synonymous? No
    Definition: a single-stranded RNA molecule that carries genetic information"
        "and from the DNA in the cell's nucleus to the cytoplasm, where proteins are synthesized"
    Are the words 'mRNA' and 'Messenger RNA' the same or synonymous? Yes
    Definition: beat or sound with a strong, regular rhythm; pulsate steadily.
    Are the words 'throb' and 'throbbing' the same or synonymous? Yes
```

## Results

![Performance of different models](/assets/img/favicons/2025-05-12-medical-definition-understanding/model_performance_comparison.png)
*Figure: Performance comparison of evaluated language models on the definition understanding task.*

### Key Takeaways
* Gemini-2.0-Flash is the best model across all metrics, Claude-3-7-Sonnet and OpenAI GPT-4.1 follow closely.
* We see that scaling laws hold and that the larger models perform better than the smaller ones.
* Unlike other popular benchmarks it seems like this seemingly simple task is far from being saturated. Gemini has a JudgeLLM accuracy of 0.821, would you trust a doctor who doesn't recognize every 5th medical term that they use 🙃?

## Future Work
* Run similar experiments to benchmark embedding models.
* Run similar experiments to benchmark models on other domains.
* Invert the definition prediction task to see if models can generate definitions given a word.

### Conclusions
One of the things about model pre-training algorithms that's bugged me for a long time is the "let's just throw everything at it" approach. I think pre-training of LLMs should have a phase specifically for **definition learning/remembering**, an LLM should never make mistakes in either defining or in recognizing definitions. I hope that this work will help push the community to think about this aspect of LLMs.

### Acknowledgements
This work was inspired by my collaboration on a tangentially related project with the AI team at [Healiom](https://healiom.com/). Special thanks to [Ash Damle](https://www.linkedin.com/in/ashdamle/) for seeding this experiment in my mind.

### Cite as
```bibtex
@article{pramodith2025medical_definition_understanding,
    author = Pramodith B,
    title = {Do LLMs know the meaning of Medical Terms?},
    journal = {pramodith.github.io},
    year = {2025},
    url = {https://pramodith.github.io/posts/2025-05-12-medical-definition-understanding/}
}
```

```
@article{pramodith2024superposition,
  title   = "Mechanistic Interpretability: What's superposition.",
  author  = "Pramodith B",
  journal = "pramodith.github.io",
  year    = "2024",
  month   = "Jul",
  url     = "https://pramodith.github.io/posts/2024-11-30-superposition/"
}
```