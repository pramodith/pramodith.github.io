---
layout: post
title: "The One Big Beautiful Blog on Group Relative Policy Optimization (GRPO)"
description: A step-by-step tutorial to code up your own GRPO Trainer.
date: 2025-06-04 10:00:00 +0000
categories: [LLM, AI, RLHF, Reasoning Models, GRPO, PPO]
tags: [LLM, AI, RLHF, Reasoning Models, GRPO, PPO]
math: true  # Enable math equations rendering
author: Pramodith B
by: Pramodith B
---
# Create your own GRPO Trainer

## Introduction
The launch of the DeepSeek family of models pushed **Group Relative Policy Optimization (GRPO)**  into the limelight among the family of Reinforcement Learning (RL) Algorithms used to train Large Language Models (LLMs). Ousting the likes of Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) from their podium. 

While PPO and DPO gained traction for aligning the responses of LLMs i.e. making them safe and useful. The GRPO
algorithm is predominantly used for training **reasoning** models. i.e. models that produce long chains of thoughts before providing a final answer.

RL training steps/loops are far more complicated than pre-training or instruction-tuning because we have multiple models (active policy, old policy, reference) etc. and each of them play a role in computing the final loss.

In this article, I'll break down GRPO and code it up step-by-step. Going through this article will enable AI/ML engineers and researchers to understand how GRPO works under the hood and all the small details and intricacies that one must consider. All the code for this tutorial is available [here](https://github.com/pramodith/llm_exploration/tree/main/simple_grpo_trainer). Here's a [link](https://arxiv.org/pdf/2402.03300) to the original paper that introduced GRPO.

## Why RL Trianing?
Supervised/Instruction fine-tuning usually require the entire answer to be laid out to a model. For e.g. a chain of thought needs to be included in the ground truth of the dataset. Reinforcement Learning allows the model to explore different strategies to solve a task on its own without any explicitly provided chain of thought.

The model essentially explores a bunch of strategies or in the case of LLMs chains-of-thought (CoT) and a reward signal rates each of these exploratory CoTs using which the model learns what the best strategies are for a task.

The potential for unconstrained exploration expands the ways the model can learn to solve a problem, since the different strategies it's exposed to isn't limited to an annotated dataset.

## The GRPO Function
The GRPO maximization function largely derives from the PPO algorithm itself, with the major difference lying in the elimination for the need of a parameterized Value Function. The Value Function is replaced by sampling a **group** of responses for each query and computing the rewards for each of the responses using a set of rules or heuristics.

In a bid to make the rules robust and meaningful, GRPO is often used on tasks with **verifiable outcomes**. i.e. we can definitively state if a given response is correct or incorrect. This is why coding, math tasks are very friendly to GRPO. Answers can be verified by creating test cases or by using simple math principles like proof by substitution even in the absence of labelled datasets.

This is also why labelled datasets become re-usable by GRPO. Algorithms like PPO/DPO require pairwise comparisions of responses which usually means an additional round of labelling (human or synthetic). However, labelled datasets already have a ground truth answer/response to the tasks and a GRPO reward function can then just compare the policy LLMs response against the ground truth answer.

### What is a policy model?
A policy model is the model that decides what action to take next in an environment. In the case of LLMs, the policy model is the model that determines which token should be generated next to best solve the query.


### Algorithm
The algorithm for GRPO from the paper is as shown below. ![GRPO Algorithm](/assets/img/favicons/2025-06-04-grpo-trainer/grpo_algorithm.png)

### Formula

The core of the algorithm lies in maximizing the GRPO objective which is formulated as:

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\Big[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)\Big] \Bigg[
\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left(
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t},
\operatorname{clip}\left(
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})},
1-\epsilon, 1+\epsilon
\right) \hat{A}_{i,t}
\right)
- \beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right]
\Bigg]
$$

That's a frightening formula with way too many symbols. So let's break it down.

#### Number of models
The formula lists out three different models:
- **Policy Model**: $\pi_\theta$ - This is the model that we are training i.e. the model whose parameters are being updated. It is the active policy.
- **Old Policy Model**: $\pi_{\theta_{\text{old}}}$ - This is the model that we are using to compute the advantage function. It is the old policy i.e. the parameters of the model are frozen for $\mu$ steps and then updated to the active policy.
- **Reference Model**: $\pi_{\text{ref}}$ - In order to ensure that the policy model doesn't go crazy because of all its exploration we use a reference model as a baseline that the policy model shouldn't deviate too much from.

 Usually the **policy and old policy are initialized as two instances of the same model**. The reference model can be almost any model that we think can be a good guide/teacher for the policy model. The reference model should be a good conversational model and serve as a good baseline for the tasks at hand.

**KL divergence** is a measure of the distance between two probability distributions. The **KL divergence** loss ensures that our new policy does not deviate too much from the reference model helping with avoiding catastrophic forgetting and reward hacking. On the flip side, the KL divergence loss also limits how different the policy model can be from the reference model. So if the reference model is bad at a task it'd be hard for the policy model to become good at the task since it cannot deviate too much from the reference model.

## Code

Alright let's get our hands dirty and start coding up our own GRPO trainer. We'll be using `PyTorch-Lightning` and HuggingFace's `transformers` library for this tutorial. Don't worry about all the other components of the equation and steps in the algorithm I'll be breaking them down and showing you how to implement them step-by-step.

For the purpose of brevity the code snippets shown in the article will be as minimal as possible. If you'd like to see the full code, please refer to the [GitHub repository](https://github.com/pramodith/llm_exploration/tree/main/simple_grpo_trainer). Note that this code only supports running the models on a single GPU.

### Loading the models
As mentioned earlier, we need three models. The policy model, the old policy model and the reference model. However, thanks to a neat trick that we'll be going through we can get away with  instantiating just two models, the reference and a policy model. We'll be using the `transformers` library to load these models.

```python
class SimpleGRPOModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto"
        )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto"
        )
        self.reference_model.eval()
```

 The code block loads the tokenizer and the models. The `policy_model` is the model that we will be training and the `reference_model` is the model that we will be using to compute the KL divergence loss. We set the `reference_model` to evaluation mode to disable dropout and other training-specific behaviors, since we'll not be updating the reference model during training.

These steps cover the first 3 steps of the algorithm. Where the policy model is initialized and the reference model is set to match the policy model.

### Data Preparation
Before we go deeper into the training loop, we need to prepare our data. We'll train a model on the [gsm8k](https://huggingface.co/datasets/openai/gsm8k) dataset. A math dataset that's been widely used for training reasoning models. We'll be using the `datasets` library from HuggingFace to load the dataset and prepare it for training.

```python
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_gsm8k_dataset():
    """
    Load the GSM8K dataset.

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset.map(extract_answer)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset
```

Let's throw in a couple of helper functions to extract the answer from the dataset and format the question in a chat template that the model can understand.

```python
def extract_answer(example: dict[str, str]):
    """
    Extract the answer from the example.

    Args:
        example (dict[str, str]): The example to extract the answer from.

    Returns:
        dict[str, str]: The example with the answer extracted.
    """
    answer_loc = example["answer"].find("### ")
    if answer_loc == -1:
        raise ValueError("Answer marker not found in example")
    example["answer"] = example["answer"][answer_loc + 4 :].replace(",", "")
    return example

def tokenize_example(example: dict[str, str], tokenizer: AutoTokenizer):
    """
    Tokenize the dataset.

    Args:
        example (dict[str, str]): The example to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        Dataset: The tokenized dataset.
    """
    system_prompt = """
    You are a helpful assistant that will use reasoning, long chain of thought, backtracking, and 
    self-reflection to answer math problems. You will respond using the following template:
    Question: [question]
    <think>
    
    </think>
    <answer>
    
    </answer>
    ## Example
    Question: Joel has 24 cookies. He ate 12 cookies. How many cookies does Joel have left?
    <think>
    Since Joel ate 12 cookies, he has 24 - 12 = 12 cookies left.
    </think>
    <answer>
    12
    </answer>

    ## Task
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {example['question']}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    example["prompt"] = prompt
    return example
```

You can also see the system prompt that we are using to ensure that the model responds with the reasoning tags `<think>` and `<answer>`.
Now we can create a `DataLoader` for the training and test datasets.

```python
def create_dataloader(
    dataset: Dataset,
    do_shuffle: bool = False,
    batch_size: int = 1,
):
    """
    Create a dataloader for the dataset.

    Args:
        dataset (Dataset): The dataset to create a dataloader for.
        do_shuffle (bool): Whether the dataset should be shuffled or not.
        batch_size (int): The batch size to use.

    Returns:
        DataLoader: The dataloader.
    """
    do_shuffle = False
    if is_train:
        do_shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle)
    return dataloader
```

Sampling from our dataloader covers **step 5** of the algorithm, 
```
5: Sample a batch D𝑏 from D
```
since it asks us to sample a batch of queries from our dataset this also corresponds to the term
$$q \sim P(Q)$$
in the equation which corresponds to sampling a query from our pool of queries.

### Sampling Multiple Responses from the Old Policy Model
```
6: Update the old policy model 𝜋𝜃𝑜𝑙𝑑 ← 𝜋𝜃
7: Sample 𝐺 outputs {𝑜_𝑖} ∼ 𝜋𝜃𝑜𝑙𝑑 (· | 𝑞) for each question 𝑞 ∈ D
```
- **Step 6** ask us to initialize the old policy model to the policy model.
- **Step 7** then asks us to sample G responses for each query using the old policy model. 

In our equation this corresponds to the expression
$$\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)$$

To do this let's update our class's constructor to include the number of responses to sample. **G** corresponds to the number of responses we want to sample for each query. All the responses from a given query belong to the same group and this is where the Group term in GRPO stems from.

```python
class SimpleGRPOModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_responses_per_example: int = 4,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_responses_per_example = num_responses_per_example
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto"
        )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto"
        )
        self.reference_model.eval()
```

Next we need to start filling out our `training_step` method that pytorch-lightning uses to run the training loop.

```python
def training_step(self, batch: Dict[str, List[str]], batch_idx: int):
    if self._step % self.num_iterations == 0:
        """
        Get the prompts and answers from the batch
        The batch is a dictionary with keys "prompt" and "answer", values are lists of strings.
        """
        inputs = self.prepare_inputs(batch)
        prompt_mask = inputs["attention_mask"]
        # Since we pad the prompts,
        # all the completions will start from the size of the padded input/prompt
        prompt_end_index = inputs["input_ids"].size(1)

        # Get the completions from the policy model
        completions, completion_ids = self.get_responses_from_policy_model(
            inputs, prompt_end_index
        )
```

Ignore the if statement for now, we'll come back to it later. The `prepare_inputs` method is a helper method that prepares the inputs for the model. It tokenizes the prompts and returns the input ids and attention masks.

We'll keep track of the the token index at which the prompt ends, we'll use this to
extract the completions from the model's output, getting rid of the input prompt and question. The completions will then be passed to our reward functions to compute the rewards for each response.

The `get_responses_from_policy_model` method is a helper method that samples G (`num_responses_per_example`) responses from the old policy model.

```python
def get_responses_from_policy_model(
        self, inputs: Dict[str, torch.Tensor], prompt_end_index: int
    ):
        with torch.no_grad():
            responses = self.policy_model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_gen_tokens,
                num_return_sequences=self.num_responses_per_example,
            )

            # Get rid of the prompt tokens in the response
            completion_ids = responses[:, prompt_end_index:]
            # Get the rewards for each response
            completions = [
                self.tokenizer.batch_decode(
                    completion_ids[
                        i * self.num_responses_per_example : (i + 1)
                        * self.num_responses_per_example
                    ],
                    skip_special_tokens=True,
                )
                for i in range(len(inputs["input_ids"]))
            ]

        return completions, completion_ids
```

Note how we're using `self.policy_model` for sampling the responses though the algorithm asked us to use the old policy, why we can do this will become clearer a little later but just keep this in mind.

We return the raw completion text and the token ids corresponding to the completions. The token ids will be used later on to compute the GRPO objective.

### Computing the Rewards for each response
Now that we have the responses from the policy model, we need to compute the rewards for each response. There's no limit on the number of reward functions that can be declared.

```
Step 8: Compute rewards {𝑟𝑖} for each sampled output 𝑜𝑖 by running the reward models
```

In our case, we'll specify two reward functions:
1. **Correctness Reward**: This reward function checks if the response is correct or not. It compares the response with the ground truth answer and returns a reward of 1 if the response is correct, 0 otherwise.
2. **Formatting Reward**: This reward function checks if the response is formatted correctly. It checks if the response contains the `<think>` and `<answer>` tags and returns a reward of 0.1 for each tag present in the response + 0.1 for the answer being a number.

The algorithm uses the term $r_{\phi}$ to denote the reward model. The word model is used loosely here, it can correspond to any function that provides a score for the model's responses.

#### Defining the Reward Functions
```python
def correct_answer_reward(answers: List[str], reference_answer: List[str]):
    """
    Calculate the correct answer reward. If the answer is correct, 
    the reward is 1.0, else 0.0.

    Args:
        answers (List[str]): The answers to calculate the reward for.
        reference_answer (List[str]): The reference answer.

    Returns:
        List[float]: The correct answer reward.
    """
    matches = []
    for answer in answers:
        match = re.match(
            r"(<think>)[\s\S]*?(</think>)[\s\S]*?(<answer>)[\s\D]*(\d+)[\s\D]*(</answer>)",
            answer,
        )
        if match:
            matches.append(match.group(4))
        else:
            matches.append(None)

    return [
        1.0
        if answer is not None and float(answer) == float(ref_answer)
        else 0.0
        for answer, ref_answer in zip(matches, reference_answer)
    ]


def format_reward(
    answers: List[str], reference_format_regex: str, per_group_reward: float = 0.1
):
    """
    Calculate the format reward. Gives a reward of `per_group_reward` for each matched group.

    Args:
        answers (List[str]): The answers to calculate the reward for.
        reference_format_regex (str): The reference format regex.
        per_group_reward (float): The reward per matched group.

    Returns:
        List[float]: The format reward.
    """
    matches = [re.match(reference_format_regex, answer) for answer in answers]
    return [
        len(
            [match.group(i) for i in range(1, len(match.groups()) + 1) if match.group(i)]
        ) * per_group_reward
        if match
        else 0.0
        for match in matches
    ]

```
These are the lines that'd be added to the `training_step` function corresponding to Step 8 in the algorithm where we compute the rewards.

```python
def training_step(self, batch: Dict[str, List[str]], batch_idx: int):
    ...
    ...
    ...
    completions_mask = self._get_completions_mask(completion_ids)
    # Compute the rewards for each response
    correct_answer_rewards, format_rewards, length_rewards = (
        self.compute_rewards(completions, batch["answer"], completions_mask)
    )
    advantage_scores = self.compute_advantage_score(
        correct_answer_rewards + format_rewards + length_rewards
)
```

The `_get_completions_mask` function masks out all the tokens that come after the very first `<eos>` token. We'll need this to compute the length of our completions.

$ |o_i| $ 
in the equation corresponds to the length of each response in the group.

```python
def _get_completions_mask(
    self, 
    sampled_responses: torch.LongTensor
) -> torch.Tensor:
    """
    Get a mask for identifying all the valid completion tokens.

    Args:
        sampled_responses: The token ids of the sampled responses/completions
    Returns:
        A masked torch tensor with 1s and 0s. 1s correspond to a valid token.
    """
    # sampled_responses: [batch_size, seq_len]
    eos_token_id = self.tokenizer.eos_token_id

    # Find the first occurrence of EOS in each response
    eos_positions = (sampled_responses == eos_token_id).int()
    # Cumulative sum along the sequence dimension
    cumsum_eos = eos_positions.cumsum(dim=1)
    # If you want strictly after (not including the EOS itself):
    after_eos_mask = cumsum_eos > 1
    after_eos_mask = ~after_eos_mask
    # We need to invert the mask to get the valid tokens
    return after_eos_mask.int()
```

We accomplish this by finding the first occurrence of the `<eos>` token in each response. The generate function pads all the responses to the same length with the `<eos>` token, so we use the cumsum function to create an incremental count for each occurence of the `<eos>` token. 

We then create a mask that identifies all the tokens that come after the first `<eos>` token. **We can then compute the length of each response by summing the mask along the sequence dimension.**

We compute the rewards by calling each of our reward functions and returning the rewards as a tuple.

```python
def compute_rewards(
    self,
    sampled_responses: List[List[str]],
    answers: List[str],
    completions_mask: torch.LongTensor,
    ):
    """
    Compute the rewards for the sampled responses.

    Args:
        sampled_responses (List[List[str]]): The sampled responses.
        answers (List[str]): The answers.
        completions_mask (torch.LongTensor): The completions mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rewards for the sampled responses.
    """
    # Repeat the answers for each response num_responses_per_example times
    answers = [
        answer for answer in answers for _ in range(self.num_responses_per_example)
    ]
    # Flatten the sampled responses
    # sampled_responses: List[List[str]] -> List[str]
    sampled_responses = [
        response[i]
        for response in sampled_responses
        for i in range(self.num_responses_per_example)
    ]
    correct_answer_rewards = correct_answer_reward(
        answers=sampled_responses,
        reference_answer=answers,
    )
    format_rewards = format_reward(
        answers=sampled_responses,
        reference_format_regex=r"(<think>)[\s\S]*?(</think>)[\s\S]*?(<answer>)[\s\D]*(\d+)[\s\D]*(</answer>)",
    )
    correct_answer_rewards = torch.tensor(correct_answer_rewards).view(
        -1, self.num_responses_per_example
    )
    format_rewards = torch.tensor(format_rewards).view(
        -1, self.num_responses_per_example
    )
    return (
        correct_answer_rewards.to(self.device),
        format_rewards.to(self.device),
    )
```

### Computing the Advantage Scores
Now that we have the rewards for each response, we need to compute the advantage scores for each response. GRPO defines the advantage score to be the standardized reward for each response. The mean and standard deviation of the rewards are computed across all responses for a given group/query.

```
Step 9: Compute 𝐴_𝑖𝑡 for the 𝑡-th token of 𝑜_𝑖 through group relative advantage estimation
```

**Step 9** of the algorithm asks us to compute the advantage scores for each response. The equation for the advantage score is given by:

$$
\hat{A}_{i,t} = \frac{r_{i,t} - \mathbb{u_i}}{\sqrt{\mathrm{Var}[r_i] + \epsilon}}
$$

Where $\mathbb{u_i}$ is the mean of the rewards for a given group. 

_Note: We compute the advantage scores over the sum of all rewards_.

```python
def compute_advantage_score(self, rewards: torch.Tensor):
    """
    Standardize the rewards. To get the advantage score of each sampled response

    Args:
        rewards (torch.Tensor): The rewards to standardize
            of shape (batch_size, num_sampled_responses).

    Returns:
        torch.Tensor: The advantage scores of shape (batch_size, num_sampled_responses).
    """
    mean_rewards = rewards.mean(dim=1).unsqueeze(1)
    std = rewards.std(dim=1).unsqueeze(1)
    advantage_scores = (rewards - mean_rewards) / (std + 1e-8)
    return advantage_scores
```

### Computing the Probability Scores of the Responses
In our equation the terms ${\pi_\theta(o_{i,t}|q, o_{i,<t})}$, ${\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$, and ${\pi_{\theta_{\text{ref}}}(o_{i,t}|q, o_{i,<t})}$ correspond to the probability score of the _t-th_ token in the _i-th_ response/completion for a given query _q_ per the current, old policy and reference model respectively.

This means that we need to extract the logit scores for all of our responses from each of our models.**Transformer models give us the logits for each token in a given input sequence in just one single forward pass through the model.** We can use this to our advantage and extract the logits for all the responses and each of the models by passing in the batch of prompt + completions to each of the models. This is exactly what we do in the `_get_completion_log_prob_scores` method below.

```python
def _get_completion_log_prob_scores(
    self,
    prompt_ids: torch.LongTensor,
    prompt_mask: torch.LongTensor,
    completion_ids: torch.LongTensor,
    completions_mask: torch.LongTensor,
    model_type: ModelType,
):
    """
    We need to obtain the logit scores of the completions from the sampled responses
    for the current-policy, old-policy and reference model.

    To do this we run a single forward pass through the model with the prompt and completion
    concatenated and get the logit scores for each of the completions.

    Args:
        prompt_ids (torch.LongTensor): The prompt ids of shape (batch_size * num_responses_per_example, seq_len).
        prompt_mask (torch.LongTensor): The prompt mask of shape (batch_size * num_responses_per_example, seq_len).
        completion_ids (torch.LongTensor): The completion ids of shape (batch_size * num_responses_per_example, seq_len).
        completions_mask (torch.LongTensor): The completions mask of shape (batch_size * num_responses_per_example, seq_len).
        is_policy_model (bool): Whether to use the policy model.
    """
    prompt_completion_input = torch.cat([prompt_ids, completion_ids], dim=-1)
    prompt_length = prompt_ids.shape[-1]
    prompt_completion_mask = torch.cat([prompt_mask, completions_mask], dim=-1)
    
    if model_type == ModelType.Active:
        self.policy_model = self.policy_model.train()
        logit_scores = self.policy_model(
            input_ids=prompt_completion_input, attention_mask=prompt_completion_mask
        ).logits
    
    elif model_type == ModelType.Old:
        self.policy_model = self.policy_model.eval()
        with torch.no_grad():
            logit_scores = self.policy_model(
                input_ids=prompt_completion_input, attention_mask=prompt_completion_mask
            ).logits
    
    elif model_type == ModelType.Reference:
        with torch.no_grad():
            logit_scores = self.reference_model(
                input_ids=prompt_completion_input,
                attention_mask=prompt_completion_mask,
            ).logits

    # If you want to use logprobs for prompt tokens, access out.prompt_logprobs
    # Logit scores are of shape (batch_size * num_responses_per_example, seq_len + 1, vocab_size)
    # We exclude the logit scores for the prompt and the last token
    # because it corresponds to the next token prediction
    logit_scores = logit_scores[:, prompt_length - 1 : -1, :]

    # Get log_probs to avoid numerical underflow/overflow
    logit_scores = logit_scores / self.temperature
    log_prob_scores = torch.log_softmax(logit_scores, dim=-1)
    # We only need to keep the logit scores corresponding to the completion tokens
    log_prob_scores = torch.gather(
        log_prob_scores, dim=-1, index=completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    return log_prob_scores.view(
        -1, self.num_responses_per_example, log_prob_scores.shape[-1]
    )

```

In the first three lines of the function, we concatenate the prompt i.e. system prompt + question with the completions. We also concatenate the attention masks for the prompt and completions so that the self-attention operation ignores all the `<pad>` tokens. We then pass this concatenated input to one of the models to get the logits for each token in the response.

Notice how we **only enable gradient computation for the active policy model**. The old policy and reference model are set to evaluation mode and we disable gradient computation for them. This is because we do not want to update the weights of these models during training. 

The astute among you might've noticed that we are using the same `policy_model` for the case of `ModelType.Old` and `ModelType.Active`. We'll come back to this in a bit.

```python
logit_scores = logit_scores[:, prompt_length - 1 : -1, :]

# Get log_probs to avoid numerical underflow/overflow
logit_scores = logit_scores / self.temperature
log_prob_scores = torch.log_softmax(logit_scores, dim=-1)
# We only need to keep the logit scores corresponding to the completion tokens
log_prob_scores = torch.gather(
    log_prob_scores, dim=-1, index=completion_ids.unsqueeze(-1)
).squeeze(-1)
return log_prob_scores.view(
            -1, self.num_responses_per_example, log_prob_scores.shape[-1]
        )
```

Since we only need the logit scores for the completion tokens, we slice the logit scores to exclude the prompt tokens and the last token. 
The last token is excluded because it corresponds to the next token prediction and we are not interested in that. 

Next,
we scale the logits by the temperature parameter. In order to get the probability scores from the logits we need to apply the softmax function, we throw in a log transformation on top of the softmax for numerical stability.

The `.logits` operation gives us the logit scores for the entire vocabulary at a given sequence positions. However, we only care about the logit scores of the tokens involved in each of the responses so, we use the `torch.gather` function to extract the logit scores for the completion tokens. Finally we reshape the logit scores to have the shape `(batch_size, num_responses_per_example, completions_length)`.

Great! we now have all the logit scores of the completions from each of our models.

### Number of Iterations and the Case of the Old Policy Model

```
4: for step = 1, . . . , M do
5:    Sample a batch D𝑏 from D
6:    Update the old policy model 𝜋𝜃(𝑜𝑙𝑑) ← 𝜋
```
**Step 6** tells us that the old policy model should be updated to the current policy for each new batch.

```
10:     for GRPO iteration = 1, . . . , 𝜇 do
11:         Update the policy model 𝜋𝜃 by maximizing the GRPO objective (Equation 21)
```
**Step 10 and 11** of the algorithm asks us to update the policy model for $\mu$ iterations. Each of these iterations/updates will be based on the **same batch of prompts and completions sampled from the old policy model.**

When we combine these two requirements together, we can see that the old policy is nothing but the **current policy for GRPO iteration = 1**. So we'll sample responses from the current policy model whenever we're at the first GRPO iteration. These same set of responses will be used to compute the GRPO loss function for the remaining $\mu - 1$ iterations.

To compute the GRPO loss we need access to the logit scores of the current and old policy, and reference model. **However, for a given batch the logit scores of the old policy and reference model will always be the same since their weights are frozen.** This means that we only need to compute the logit scores for those two models at the **very first GRPO iteration and we can re-use them for all subsequent GRPO iterations.**

Let's take a look update the `training_step` method based on all of this information.

```python
def training_step(self, batch: Dict[str, List[str]], batch_idx: int):

    """
    The num_iterations parameter is used to specify how many times the policy model
    will be updated for a given batch of prompts.

    Since the old policy is used to generate responses and both the old policy and reference
    model are not updated via backprop. The responses/logit scores from the old policy and reference
    model will be the same for all iterations. So we only generate these values/tensors once
    and use them for all iterations.

    Whereas we get the logit scores for the policy model for each iteration. **This also means that we
    don't need to have a separate instance for the old policy model. At all iterations the response from the 
    `old policy` is the same as the current policy at iteration 0.**
    """
    if self._step % self.num_iterations == 0:
        inputs = self.prepare_inputs(batch)
        # Get the prompts and answers from the batch
        # The batch is a dictionary with keys "prompt" and "answer"
        # and values are lists of strings.
        prompt_mask = inputs["attention_mask"]
        # Since we pad the prompts,
        # all the completions will start from the size of the padded input/prompt
        prompt_end_index = inputs["input_ids"].size(1)

        # Get the completions from the policy model
        completions, completion_ids = self.get_responses_from_policy_model(
            inputs, prompt_end_index
        )

        completions_mask = self._get_completions_mask(completion_ids)

        correct_answer_rewards, format_rewards = (
            self.compute_rewards(completions, batch["answer"], completions_mask)
        )

        # Log total rewards per step
        average_rewards = (
            (correct_answer_rewards + format_rewards).mean().item()
        )
        advantage_scores = self.compute_advantage_score(
            correct_answer_rewards + format_rewards
        )
        # Repeat the prompts for each response
        prompt_ids = inputs["input_ids"].repeat_interleave(
            self.num_responses_per_example, dim=0
        )
        prompt_mask = inputs["attention_mask"].repeat_interleave(
            self.num_responses_per_example, dim=0
        )
        self.cache["advantage_scores"] = advantage_scores
        self.cache["prompt_ids"] = prompt_ids
        self.cache["prompt_mask"] = prompt_mask
        self.cache["completion_ids"] = completion_ids
        self.cache["completions_mask"] = completions_mask
        
        # If we have just one iteration, the old policy's logit will always be
        # the same as the current policy's logit.
        if self.num_iterations > 1 and self._step % self.num_iterations == 0:
            # for the old policy model as well.
            old_policy_logit_scores = self._get_completion_log_prob_scores(
                self.cache["prompt_ids"],
                self.cache["prompt_mask"],
                self.cache["completion_ids"],
                self.cache["completions_mask"],
                model_type=ModelType.Old,
            )
        else:
            # The old policy model is the same as the current policy model so the outputs would
            # be the same.
            old_policy_logit_scores = None
        self.cache["old_policy_logit_scores"] = old_policy_logit_scores

        # Compute the forward pass with gradient calculation disabled.
        reference_logit_scores = self._get_completion_log_prob_scores(
            self.cache["prompt_ids"],
            self.cache["prompt_mask"],
            self.cache["completion_ids"],
            self.cache["completions_mask"],
            model_type=ModelType.Reference,
        )
        self.cache["reference_logit_scores"] = reference_logit_scores

    # Compute the forward pass with gradient calculation enabled.
    policy_prob_scores = self._get_completion_log_prob_scores(
        self.cache["prompt_ids"],
        self.cache["prompt_mask"],
        self.cache["completion_ids"],
        self.cache["completions_mask"],
        model_type=ModelType.Active,
    )

    if self.cache["old_policy_logit_scores"] is None:
        old_policy_logit_scores = policy_prob_scores.detach()
        self.cache["old_policy_logit_scores"] = old_policy_logit_scores

    loss = self.compute_grpo_loss(
        policy_prob_scores,
        self.cache["old_policy_logit_scores"],
        self.cache["reference_logit_scores"],
        self.cache["advantage_scores"],
        self.cache["completions_mask"].view(
            -1, self.num_responses_per_example, self.cache["completion_ids"].shape[-1]
        ),
    )

    self.log_dict(
        {
            "train_loss": loss,
            "train_average_rewards": average_rewards,
        },
        on_step=True,
        on_epoch=False,
        prog_bar=True,
    )
    return loss
```

We introduce a new cache dictionary to store the intermediate results that we'll need to compute the GRPO loss. This cache will store the prompts, completions, logit scores and advantage scores for each batch. All of these cached values will be used in subsequent iterations of the GRPO training loop.

The first if condition checks if we're at the first iteration of the GRPO training loop
```python
if self._step % self.num_iterations == 0:
```

If we are, we prepare the inputs, sample the responses from the policy model, compute the rewards and advantage scores, and compute the logit scores for the reference model. All of these values are stored in the cache dictionary.

For all subsequent iterations, we only compute the logit scores for the policy model and use the cached values for the old policy model and reference model.

### Simulating Iterations
To help with consolidating ones understanding let's simulate two iterations.

#### Iteration 0
1. Sample Policy Model for responses and logit scores.
2. Sample Reference Model for logit scores of those responses.
3. At Iteration 0, $\pi_\theta = \pi_{\theta\text{old}}$ so we re-use the logit scores from step 1 for the old and current policy.
4. Store responses, logit scores of the **policy model** as `old_policy_logit_scores` and logits of reference model in the cache.
5. Compute GRPO Objective and update the weights of the current policy.

#### Iteration 1
1. Load responses from cache.
2. Compute logit scores of the loaded responses using the updated i.e. current policy model.
3. Load reference model logit scores from cache.
4. Load the logit scores of the policy model from Iteration 0 i.e. the one we saved as `old_policy_logit_scores` to serve as the **old policy's** logit scores.
5. Compute GRPO Objective and update the weights of the current policy.

**So by caching the responses and logit scores of the policy model at Iteration 0 we can get rid of the need for a separate instance to represent the old policy.**

#### Replicating batches in our dataloader.

In order to make sure that our dataloader behaves as desired when we have multiple GRPO iterations, we need to duplicate each record in our dataset the same number of times as the number of GRPO iterations. We accomplish this by the helper function `repeat_row_n_times`

```python
def repeat_row_n_times(dataset: Dataset, n: int):
    return Dataset.from_list([row for _ in range(n) for row in dataset]
)
```

## Computing the GRPO Loss
Alright time to dig into the GRPO loss function. This section is going to be fairly math heavy, so buckle up. I promise it won't be very complicated/

I've been using the terms _GRPO Objective and GRPO Loss_, let me clarify the difference. **The GRPO Loss is the negative of the GRPO Objective.** In Reinforcement Leanring the goal is to maximize a score in this case the goal is to **maximize the GRPO Objective**. However, most ML frameworks work by minimizing a loss, **so we negate the GRPO objective** call it the GRPO loss and minize it.

The GRPO objective can be broken down into two key terms:

$$
\text{GRPO}_{objective} = \text{PolicyScore} - \beta * \text{KLDiv}
$$

$\beta$ is a hyper-parameter. 

### KL-Divergence
The KL divergence loss computed as

$$
\mathbb{D}_{\mathrm{KL}}\left[\pi_\theta \| \pi_{\mathrm{ref}}\right] = 
\frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1
$$

The first term is the ratio of probability scores of a given token in the response between the reference model and the current policy model. The second term is the log of the ratio of probability scores. 

Since we have the logit scores for each of the models and all the repsonses we can code this up as follows:

```python
def compute_grpo_loss(
        self,
        policy_logprob_scores: torch.Tensor,
        old_policy_logprob_scores: torch.Tensor,
        reference_logprob_scores: torch.Tensor,
        advantage_scores: torch.Tensor,
        completions_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the GRPO loss.

        Args:
            policy_logprob_scores (torch.Tensor): The probability scores from the policy model
                of shape (batch_size, num_responses_per_example, completions_seq_len).
            old_logpolicy_prob_scores (torch.Tensor): The probability scores from the old policy model
                of shape (batch_size, num_responses_per_example, completions_seq_len).
            reference_logprob_scores (torch.Tensor): The probability scores from the reference model
                of shape (batch_size, num_responses_per_example, completions_seq_len).
            advantage_scores (torch.Tensor): The advantage scores
                of shape (batch_size, num_responses_per_example).
            completions_mask (torch.Tensor): The mask for the completions
                of shape (batch_size, num_responses_per_example, completions_seq_len).
        Returns:
            torch.Tensor: The GRPO loss.
        """
        # GRPO uses a custom forumation of KL divergence loss that's always positive
        ref_policy_logprob_diff = reference_logprob_scores - policy_logprob_scores
        per_token_kl_div_loss = (
            torch.exp(ref_policy_logprob_diff) - ref_policy_logprob_diff - 1
        )
```

Remember that the scores that we computed in the `_get_completion_log_prob_scores` are the logs of the probability scores. So our implementation needs to transform the log probs back to their raw probability scores, which can be done via a simple `exp` operation.

$$
\exp^{log(p)} = p
$$

### Policy Socre
The policy score is

$$
\min \left[
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t},
\;
    \operatorname{clip}\left(
    \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})},
    1-\epsilon, 1+\epsilon
    \right) \hat{A}_{i,t}
\right]
$$

$\epsilon$ is another hyper-parameter. On the whole the policy score tries to make sure that the current policy model produces higher probability scores for the tokens which lead to an outcome where the advantage/rewards are high and lower scores for the tokens which resulted in an outcome where the rewards were low.

We perform clipping, in order to make sure that the policy model doesn't drastically drift away from its previous version and avpid reward hacking and/or catastrophic forgetting.

The ratio of probability scores between the current policy and the old policy is a common term.

$$
\text{X} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}
\;
$$

```python
policy_ratio = torch.exp(policy_logprob_scores - old_policy_logprob_scores)
```

We then need to perform an upper and lower clipping of the policy ratio.

$$
\text{Y} = \operatorname{clip}\left(
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})},
1-\epsilon, 1+\epsilon
\right)
$$
```python
clipped_policy_loss = torch.clamp(
    policy_ratio, 1 - self.epsilon, 1 + self.epsilon
)
```
Finally we need to compute the element-wise minimum of the policy ratio and the clipped policy ratio, multiplied by the advantage scores.

It's important to note that the same advantage scores are applied to all the tokens for a given response.
$$
\text{min}(X*\hat{A}_{i,t}, Y*\hat{A}_{i,t})
$$

```python
policy_score = torch.minimum(
    policy_ratio * advantage_scores.unsqueeze(-1),
    clipped_policy_loss * advantage_scores.unsqueeze(-1),
)
```

The GRPO objective corresponds to a value that needs to be maximized. However, optimizers in common ML frameworks operate on minimizing a loss.
So we simply negate the the GRPO objective. **Minimizing a negative of a value is the same as maximizing the original value.**

At this point it's good to note that the shape of the `grpo_loss` tensor and the rest of the tensors used are of shape `(batch_size, num_responses_per_example, max_completion_length_in_batch)`. So each batch index would contain all the completions of a give query/group.

```python
grpo_loss = -1.0 * (policy_score - self.beta * per_token_kl_div_loss)
grpo_loss = (grpo_loss * completions_mask).sum(dim=-1)
grpo_loss /= completions_length
grpo_loss = grpo_loss.mean()
return grpo_loss
```

The **completions_mask** will zero out the scores for the pad/eos tokens.

$ \sum_{t=1}^{|o_i|}$ 
tells us to sum up all the scores of each of the tokens in a response and then normalize the loss by the length of the completions, allowing us to negate any length bias.

$ \Bigg[\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|}\Bigg] $ 
tells us that we need to average (sum up scores across all groups of the batch and then divide by the number of groups) the scores across all the groups. This is accomplished by the last line where we compute the mean.

## Training the Model
That's it! We now have all the components needed to train our own reasoning model using GRPO. We can skip Step 12 of the algorithm since our reward models are not parameterized and are just rule based functions.

### Benchmarking a Model
Our experiments were run on a NVIDIA A100 GPU with 40GB of memory. We use HuggingFace's `SmolLM2-1.7B-Instruct` as our policy and reference mdoel. To establish a baseline for the model before any RL training we sample 8 responses for each question in the test split of the `gsm8k` dataset and compute the average rewards for each response.

We then train a reasoning model using our Pytorch-Lightning implementation. The model is trained for 200 policy model updates, with a gradient accumulation of 4, `num_iterations`($\mu$) set to 1 and a batch size of 1. This means that we'll go through 800 questions and update the policy model every 4 questions. 

#### Hyper-parameters
We use the same set of sampling hyper-parameters for both benchmarking and training the model, most of these also correspond to the default values used in the paper and/or the `TRL` [library](https://github.com/huggingface/trl).

```yaml
- train_temperature: 0.9
- test_temperature: 0.7
- top_p: 0.9
- top_k: 50
- max_gen_tokens: 300
- num_responses_per_example: 8
- beta: 0.04
- epsilon: 0.2
- train_batch_size: 1
- gradient_accumulation_steps: 4
- learning_rate: 5e-5
- scheduler: linear_lr
- warmup_steps: 0
- gradient_clipping: 1.0
```

#### Results

| Model Name                                         | Mean Accuracy Rewards | Mean Format Rewards |
|----------------------------------------------------|----------------------|----------------------|
| HuggingFaceTB/SmolLM2-1.7B-Instruct-Baseline       | 0.2927               | 0.2573               |
| HuggingFaceTB/SmolLM2-1.7B-Instruct-Pramodith-GRPO | 0.4355               | 0.4248               |

You can see that the training helped the model improve both in terms of the average accuracy as well as how well it sticks to the requested response template.

## Curious Case of Number of Iterations = 1
Almost all tutorials, the default setting in TRL and even the original paper report setting the number of iterations i.e. $\mu$ = 1.

In my opinion this could be hindering the quality of the loss signal. Let me explain why. 

Let's just focus on the $PolicyScore$ term.

$$
\min \left[
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t},
\;
    \operatorname{clip}\left(
    \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})},
    1-\epsilon, 1+\epsilon
    \right) \hat{A}_{i,t}
\right]
$$


The policy loss involves computing the ratio of probabilites for a given completion token between the current and old policy. However, as discussed earlier for the first GRPO iteration, the old and current policies are exactly the same.

This means that $\pi_\theta(o_{i,t}) = \pi_{\theta{\text{old}}}(o_{i,t})$

Which in turn means that 
$$
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} = 1.0
$$

When we plug this into the expression corresponding to the Policy Loss
we'll see that 

$$
\text{PolicyLoss} = -\text{min}(1*\hat{A}_{i,t}, clip(1, 1-\epsilon, 1+\epsilon)*\hat{A}_{i,t})
$$

The clip term will always reduce to 1 since

$$
1-\epsilon < 1 < 1+\epsilon
$$

This means that
$$
\text{PolicyLoss} = -\hat{A}_{i,t}
$$

But when we average the loss over the groups **it'll always end up being 0.0.**

### Policy Loss is always 0
Okay so we've established that the $PolicyLoss$ is always equal to the advantage score for each token of the completion.

Since all the tokens have the same score, the mean PolicyLoss score for a given response is the same advantage score.

$$
\text{PolicyLoss}_{completion\_i} = -\frac{1}{T}\sum_{t=0}^T\hat{A}_{i,t} = -\hat{A}_{i,t},   \text{ where T} = |o_i|
$$

Now let's say that we have 4 completions per group and our batch size is 1. Our PolicyLoss Tensor would now look like

$$
\text{PolicyLoss}_{group} = -[\hat{A}_{0}, \hat{A}_{1}, \hat{A}_{2}, \hat{A}_{3}]
$$

So our group's policy loss is exactly the same as our advantage scores for a given group. Now let's recall that the formula for the Advantage Scores is

$$
\hat{A}_{i,t} = \frac{r_{i,t} - \mathbb{u_i}}{\sqrt{\mathrm{Var}[r_i] + \epsilon}}
$$

This is the standardization formula. This [formula](https://math.umd.edu/~mboyle/courses/131f15/stand.pdf) takes in a vector and transforms it into a vector **with a mean of 0**!!!

So

$$
\text{PolicyLoss} = -\frac{1}G * \sum PolicyLoss_{group}
$$
$$
\text{PolicyLoss} = -mean(PolicyLoss_{group}) = 0
$$

Which means that our total final loss would just end up being the KL-divergence between the policy and reference models.

$$
\text{Loss} = 0 + \beta * \text{KLDiv}
$$

If you've followed any TRL/Unsloth tutorial and trained a reasoning model your loss is just the KL-divergence loss because the default $\mu$ is 1 and the tutorials don't update it.
To prove this to yourself further, take any Unsloth notebook or a TRL tutorial and set the `beta` param to 0.0 and then train a reasoning model using the `GRPOTrainer` you'll see that the loss is always 0. **Since beta * KL = 0 and like we've shown the Policy Loss is always 0 too.**

### More Pecularities KL-div loss is 0 at step 0
When the reference model and the policy model are first initialized at the very first training step they have the exact same set of weights/parameters. This means that the KL-div loss is 0. Since 

$$
\frac{\pi_{\theta_{\text{ref}}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} = 1.0
$$

Which means

$$
\text{KLDiv} =
\mathbb{D}_{\mathrm{KL}}\left[\pi_\theta \| \pi_{\mathrm{ref}}\right] = 
\frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1
$$

$$
\text{KLDiv} = 1 - log(1) - 1 => 1 - 0 - 1 = 0
$$

Since both the Policy Loss and GRPO Loss are 0 at the first training step **the GRPO loss will be 0.0**. To prove this to yourself log out the loss at step 0 using the `GRPOTrainer` from TRL it'll always be 0.


### How on earth does it all work then?
We're now faced with two observations if the number of iterations i.e. $\mu$=1
1. The Loss at step 0 of the algorithm will always be 0.
2. The Policy Loss will always be 0 (since there's only one iteration for each batch/query), which means that our **loss value isn't affected by our rewards**.

Now, having a loss function designed around **verifiable rewards** only to find out that the rewards don't contribute to the loss is quite befuddling.

#### Parameter Updates Leverage the Rewards
The first thing to realize is that unlike more commonly used loss functions like CrossEntropy, Mean Square Error, KL divergence etc. the GRPO objective doesn't have a lower bound of 0. The GRPO objective can be both negative and positive.

$$
\text{GRPO}_{loss} = - \text{PolicyScore} + \beta*\text{KLDiv}
$$
Range of KLDiv is **[0, inf]** and the range of the Policy Loss is going to be 

$$
[(1-\epsilon) * \text{Min}(\hat{A}_{i,t}),(1+\epsilon) * \text{Max}(\hat{A}_{i,t})]
$$

This means that our optimization process doesn't stop when the loss is 0 unlike Cross Entropy/MSE etc., our loss value can go lower we can continue minimizing it!

The range depends on the Advantage scores which can be negative or positive.

Next thing to bear in mind is that just because the value of the loss function is 0 **doesn't mean that the gradients of the loss with respect to a parameter would be 0 too.**

The deriviative of the PolicyLoss at step=0 and the first iteration with respect to the models parameters would be:

### Policy Loss Derivative

The policy loss function is:

$$\mathcal{L}_{\text{policy}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clipped term}\right)$$

At step 0 when $\mu = 1$, we have $\pi_\theta = \pi_{\theta_{\text{old}}}$, so the probability ratio equals 1 and the min operation selects the first term (since clipping doesn't affect a ratio of 1).

The derivative with respect to $\theta$ is:

$$\nabla_\theta \mathcal{L}_{\text{policy}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_{i,t} \cdot \nabla_\theta \left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}\right)$$

Since $\pi_{\theta_{\text{old}}}$ is constant with respect to $\theta$:

$$\nabla_\theta \mathcal{L}_{\text{policy}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_{i,t} \cdot \frac{1}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \cdot \nabla_\theta \pi_\theta(o_{i,t}|q, o_{i,<t})$$

Using the identity $\nabla_\theta \pi_\theta = \pi_\theta \cdot \nabla_\theta \log \pi_\theta$:

$$\nabla_\theta \mathcal{L}_{\text{policy}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_{i,t} \cdot \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \cdot \nabla_\theta \log \pi_\theta(o_{i,t}|q, o_{i,<t})$$

At step 0, since $\pi_\theta = \pi_{\theta_{\text{old}}}$, the ratio equals 1:

$$\nabla_\theta \mathcal{L}_{\text{policy}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_{i,t} \cdot \nabla_\theta \log \pi_\theta(o_{i,t}|q, o_{i,<t})$$

We can see that the derivative of $\mathcal{L}_{\text{policy}}(\theta)$ gets a signal from the Advantage scores and thereby the reward functions.

The case of $\mu$ = 1 is still interesting, because there's no point in having an old policy. In some future work I'll consider experimenting with different values of $\mu$.


## Conclusion
If you've made it to the end of this blog, you will have mastered the GRPO algorithm. You know what all the scary symbols mean and can now explain them if someone were to wake you up in the middle of the night.

You know all the intricacies of how to implement it and the neat tricks that can be used to make it run efficiently. 

You can also appreciate how the training loop of an RL algorithm is quite different from that of traditional pre-training or instruction tuning/SFT.

Until the next time, take care and be kind.

## Cite as
```bibtex
@article{pramodith2025_grpo_tutorial,
    author = Pramodith B,
    title = {"The One Big Beautiful Blog on Group Relative Policy Optimization (GRPO)},
    journal = {pramodith.github.io},
    year = {2025},
    url = {https://pramodith.github.io/posts/2025-06-04-grpo-trainer/}
}
```