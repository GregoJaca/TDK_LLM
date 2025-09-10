This file describes the structure and part of the content of the paper.

Abstract

Introduction and Motivation

Related Work

Preliminaries

- Nonlinear Dynamics and Chaos
-- Definition. determinism. Trajectories in Phase Space. numerical solution.
-- Features of chaotic systems: Sensitivity to initial conditions (SIC). Divergence of trajectories (distance time series) and lyapunov exponents. periodicity and recurrence and recurrence plots. measurement error, uncertainty and predictability. attractors fractality and dimension.
-- chaos vs randomness


- Large Language Models (LLMs)
-- definition
-- tokenization, embeddings, hidden states and their correspondence
-- layers, mlp, nonlinearity, attention, context
-- inference: decoding stategies. Temperature and probabilistic sampling. greedy sampling and determinism. data obtained from inference: tokens, text, hidden states. briefly distilling and floating point precision
-- training, interpretability, where is knowledge and data stored


- Analogy / Similarity between LLMs and Nonlinear chaotic systems (a table and brief text)

<start table>
Chaotic System - LLM
phase space trajectory - sequence of text, tokens, or hidden states
numerical solution - inference
uncertainty in measurement - float point errors and distillation
nonlinear coupling - feedback and chaos
<end table>

-- Limitations: llm trajectories are discontinuous and the step size is fixed (single token). you can't have higher resolution. model is trained to predict tokens, not with the explicit of being interpretable. sampling collapses the hidden state at the last layer and gets next token, this step is quite discontinuous and has a big impact.

Methods

- initial conditions generation: get a prompt. tokenize and embed, get a single tensor shape (n_tokens, hidden_dim). then generate multiple copies and add a random direction fixed magnitude disturbance. get a list of tensors of len(n_initial_conditions)

- Distance Metrics (for each  metric explain their definition with math formula, intuitive meaning: what are they measuring, and )7

-- Between single vectors: you get a single scalar for each pair of vectors, and a sequence of 
--- cosine similarity
--- cross correlation

-- Between full trajectories (you get a single scalar for each pair of trajectories)
--- dynamic time wrapping dtw
--- hausdorff
--- frechet
--- svd eigenvector ranking + a deviation metric: sum(cos_distance(of each matching pair of eigenvectors) xor rms (mean abs) of the ranking - expected (which is linear)

-- sliding windows, aggregation, pooling
-- Divergence and Lyapunov

- Recurrence Plots 
- Pointwise and Correlation dimensions
- Clustering

Results

Here we have the data from the hidden states or from the embedded text. Also we have a lot of different metrics

- Divergence of trajectories and SIC
-- Threshold of minimum intial distance in order to get divergence. Mention implication of sensitivity to initial conditions / measurement error and floating point errors and distillation.


Discussion and Conclusions

- The dynamical effect of LLM sampling

Acknowledgements

References

Appendix





