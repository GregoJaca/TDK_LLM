This file describes the structure and part of the content of the paper.

Abstract

Introduction and Motivation

Related Work (I've seen many papers in ai have a tradition of summarizing recent relevant lit in the beggining) (Also these are the papers that I'll cite for most of the conceptual underpinning) (I will ofc add more citations when needed for specific claims)

https://arxiv.org/pdf/2503.13530 
the LLMs reasoning ability stems from a chaotic process of dynamic information extraction in the parameter space.

https://www.sciencedirect.com/science/article/abs/pii/089360809090050U 



https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ 
reproducibility and determinism is now desirable in LLMs. In part because it provides a clear (not fuzzy) signal to train LLMs with RL. For this, randomness from GPU batching variance and some floating point errors are being controlled. The sensitivity of LLMs to small numerical errors is relevant, and I talk about it.
Batch-invariant kernels unlock true reproducibility
G: this blog (for some reason) has been quite popular lately. it is great


G: On the other hand generating varied output (not deterministic and fixed) is desireable some times (like then wanting to make synthetic datasets or new images or text (paraphrasing)).

https://arxiv.org/pdf/2502.15208 LLMs have attractors and stable configurations when using LLM as paraphrasers. 2 period attractor cycles.

https://journals.plos.org/complexsystems/article?id=10.1371/journal.pcsy.0000027 
The generated image sequences occupy a more limited region of the image space compared with the original training dataset.
Positive Lyapunov exponents estimated from the generated trajectories confirm the presence of chaotic dynamics, with the Lyapunov dimension of the attractor found to be comparable to the intrinsic dimension of the training data manifold.
G: CycleGAN makes output in attractor or stable configs reducing diversity (compared to train data)


G: Interpretability of RNNs from nonlinear dynamics description
https://barak.net.technion.ac.il/files/2012/11/sussillo_barak-neco.pdf
fixed points, both stable and unstable, and the linearized dynamics around them, can reveal crucial aspects of how RNNs implement their computations.
the mechanisms of trained networks could be inferred from the sets of fixed and slow
points and the linearized dynamics around them.

https://arxiv.org/pdf/2410.02536
Both uniform and periodic systems, and often also
highly chaotic systems, resulted in poorer downstream performance, highlighting
a sweet spot of complexity conducive to intelligence. 
G: completely chaotic or completely uniform periodic systems are not intelligent. initelligence is at the edge between the two.


Preliminaries

- Nonlinear Dynamics and Chaos
-- Definition. determinism. Trajectories in Phase Space. numerical solution.
-- Features of chaotic systems: Sensitivity to initial conditions (SIC). Divergence of trajectories (distance time series) and (discrete) lyapunov exponents. periodicity and recurrence and recurrence plots. measurement error, uncertainty and predictability. attractors fractality and dimension.
-- chaos vs randomness

- Large Language Models (LLMs)
-- definition
-- tokenization, embeddings, hidden states and their correspondence
-- layers, mlp, nonlinearity, attention, context window
-- inference: decoding stategies. Temperature and probabilistic sampling. greedy sampling and determinism. data obtained from inference: tokens, text, hidden states. briefly distilling and floating point precision
-- training, interpretability, where is knowledge and data stored


- Analogy / Similarity between LLMs and Nonlinear chaotic systems (a table and brief text)

<start table>
Chaotic System - LLM
phase space trajectory - sequence of text, tokens, or hidden states
numerical solution - inference
uncertainty in measurement - batching variance, float point errors, and distillation
nonlinear coupling - attention
<end table>

we define trajectory for LLM in a broad sense as the sequence of outputs of the LLM (either in token text space or in hidden state representation)
very naturally, inference gives you the whole trajectory, analogous to solving numerically the differential equation

emphasize that in order for chaos to exist, there must be nonlinear coupling between the variables in the differential equation. In LLMs attention introduces a nonlinear feedback coupling.

!! also there is nondeterminism from batching variance


-- Limitations and Differences: llm trajectories are discontinuous and the step size is fixed (single token). you can't have higher resolution. model is trained to predict tokens, not with the explicit of being interpretable. sampling collapses the hidden state at the last layer and gets next token, this step is quite discontinuous and has a big impact on the divergence. High dimensionality of data.

Methods

- initial conditions generation: get a prompt. tokenize and embed, get a single tensor shape (n_tokens, hidden_dim). then generate multiple copies and add a random direction fixed magnitude disturbance. get a list of tensors of len(n_initial_conditions)

- working directly with hidden states or taking text and using a sentence embedder (pretrained opensource) to obtain a tensor (list of vectors). noth are nice. the embedder captures context and smoothens data.

- Distance Metrics (for each  metric explain their definition with math formula, intuitive meaning: what are they measuring)

-- Between single vectors: you get a single scalar for each pair of vectors, and a sequence of 
--- cosine similarity
--- cross correlation

-- Between full trajectories (you get a single scalar for each pair of trajectories)
--- dynamic time wrapping dtw
--- hausdorff
--- frechet
--- svd eigenvector ranking + a deviation metric: sum(cos_distance(of each matching pair of eigenvectors)) xor rms (mean abs) of the ranking - expected (which is linear)

-- sliding windows, aggregation, pooling
-- Divergence and Lyapunov

- Recurrence Plots (RP)
-- intuition and what do they tell us.Qualitative structures visually identifiable.
--- RP of typical perodic, chaotic, and white noise systems 
-- RQA
- Pointwise and Correlation dimensions
- Clustering

Results

Here we have the data from the hidden states or from the embedded text. Also we have a lot of different metrics

- Divergence of trajectories and SIC
-- Different metrics paint a similar picture
-- Threshold of minimum intial distance in order to get divergence. Mention implication of sensitivity to initial conditions / measurement error to -> floating point errors and distillation and batching variance.
--- Maybe trajectories dont diverge for very small initial perturbations (an explanation is that the sampling for autoregression kills it)


- Recurrence Plots
-- Similarity to chaotic systems
-- Differences between layers
-- relate RP regions and structures to text output


- Pointwise and Correlation dimensions
-- good fit. fractal
--- difference between layers. last layers paint a better picture. this is quite important or trivial. It could be that at first layer, all the hidden states for a same repeated token are identical and thus you get a lot of repeated vectors and not a fine fractal structure. on the contrary, when using last layer, the hidden states corresponding to the same token will be different. this gies much more variety and fractal.
-- low dimension. enigma. pca explained variance dimension.
-- limitations. unsufficient data

- Clustering


Discussion and Conclusions

- The dynamical effect of LLM sampling. Does it break the exponential chain and reduce divergence to sub-exponential



Acknowledgements

References

Appendix

Intuition about metrics. Effect of hidden states vs embeddings and the metrics used for distance divergence calculations. Use RP for each combination to give intuition of the effect.
Embedding and some distance techniques smoothe out, pool, capture context





