---
layout: layout.njk
title: "Towards a Physics Foundation Model"
---

<div class="authors">
Florian Wiesner, Matthias Wessling, Stephen Baek
</div>

<div class="gif-grid-3x3-nospace">
<img src="assets/nextstep/euler_peri_gt_channel_3.gif" alt="Euler equations with periodic boundary conditions">
<img src="assets/nextstep/turbulent_gt_channel_2.gif" alt="Turbulent flow simulation">
<img src="assets/nextstep/twophase_gt_channel_2.gif" alt="Two-phase flow dynamics">
<img src="assets/nextstep/rb_gt_channel_2.gif" alt="Rayleigh-Bénard convection">
<img src="assets/nextstep/supersonic_gt_channel_1.gif" alt="Supersonic flow simulation">
<img src="assets/nextstep/shear_gt_channel_3.gif" alt="Shear flow dynamics">
<img src="assets/nextstep/obj_peri_gt_channel_3.gif" alt="Object interaction with periodic boundaries">
<img src="assets/nextstep/euler_open_gt_channel_2.gif" alt="Euler equations with open boundary conditions">
<img src="assets/nextstep/heat_gt_channel_2.gif" alt="Heat transfer simulation">
</div>

Paper: [arXiv](https://arxiv.org/abs/2509.13805)

## TLDR

In our paper "Towards a Physics Foundation Model", we introduce a transformer-based model designed to learn multiple physical systems. Our goal is to create a unified framework — a Physics Foundation Model — that can understand and predict physical phenomena across multiple scales and domains. Importantly, such a foundation model **must be
capable of generalizing to new physical systems and conditions not seen during training**. We compare our model's performance against traditional physics-surrogate models and demonstrate its ability to generalize to unseen scenarios.


## A Foundation Model for Physics — Why?

> "A Foundation Model exhibits wide generalization capability – without retraining"

Large Language Models (LLMs), especially their newest version of Reasoning Models, are extremely versatile. They can write poems, give summaries, translate, code and solve math problems. This is not only due to their vast training data, but also their ability of in-context and zero-shot learning; they can adapt to new tasks without any additional training, simply by following their instructions. Think for example of this prompt:

```
Take this paper and
- Translate it to German.
- Replace all em-dashes with parentheses.
- Paraphrase the last section.
- Check all math formulas.

Return Latex code!
```
This particular task (the paper and my instructions) is something the model has never seen before, yet it can solve it fantastically well.

**We want something similar for physics!** We want a model that learned all possible physical principles and can apply them to new scenarios, without any retraining.
Importantly, we are talking about a replacement of traditional physics-simulators, not LLMs. These models would not output physical equations (text), but physical fields (numerical data), for example the velocity field of a fluid flow.
Current physics-aware machine learning (PAML) models are typically narrow models, and for good reason, or actually two reasons:

1. Data is scarce, since simulations and experiments are expensive as hell. Espcially 3D simulations take weeks to months on a supercomputer. Also, the physical world is extremely diverse, and it is almost impossible to cover all scenarios.
2. Predicting physics is extremely hard, even for numerical solvers. Physics simulations are done time-step by time-step, and small errors accumulate quickly. Machine learning models typically struggle with these errors when predicting long-horizon phenomena.

## The General Physics Transformer (GP<sub>hy</sub>T)

The General Physics Transformer is our first step towards a Physics Foundation Model. Its abbreviation GP<sub>hy</sub>T obviously refers to GPT, the Generative Pre-trained Transformer, by OpenAI. We think of GP<sub>hy</sub>T as the version 0.1 and hopefully by the time we reach version 3.5 or 4, we will have a true Physics Foundation Model, equally successful as its NLP counterpart. For now, we focus on learning a limited set of physical systems (fluid dynamics), since it is the domain with the most amount of data available.

GP<sub>hy</sub>T is a transformer-based model designed to learn universal physical principles from a sequence of time steps, a "physics prompt". This prompt allows the model to understand the current state of a physical system **without additional input**. Which boundary conditions, which spatial dimensions and which physics are present, is all encoded in the prompt. The model then predicts the next time step, and can be applied autoregressively to predict long-term dynamics.

The specific architecture is shown below. The model encodes multiple physical fields (e.g. velocity, pressure, temperature) at multiple time steps using a linear encoder (tokenizer) into spatiotemporal patches. Additionally, we compute the spatial and temporal derivatives of these fields using finite differences. This helps the model to better identify the underlying physics. The encoded space-time patches are then processed by a standard transformer architecture. Importantly, the transformer computes the partial derivatives wrt. time ($\frac{\partial X}{\partial t}$), which are then integrated using a numerical integrator. The ML community already knows that predicting differences rather than absolute values is much easier for neural networks (resnets, skip connections, ...), so this is not really surpising. Here, we take this idea to the next level by predicting physical time derivatives.

<img src="assets/arch.png" alt="Architecture diagram" class="arch-image">


## Results on Known Physics

We first evaluated GP<sub>hy</sub>T on the known physical systems it was trained on and compared against strong baselines such as a standard UNet and the Fourier Neural Operator (FNO), one of the best physics-aware models for single physics tasks. GP<sub>hy</sub>T outperforms all baselines on all systems, often by a large margin. Interestingly, GP<sub>hy</sub>T with 10M parameters still outperforms much larger FNO or UNets, showing the efficiency of the transformer architecture. Below are some example predictions of GP<sub>hy</sub>T (bottom row) against the ground truth (top row). You almost cannot tell the difference and for the periodic systems like shear flow (left) and euler equations (center-right), the boundaries are almost invisible.

  <div class="gif-grid-4x2-nospace">
      <img src="assets/nextstep/shear_gt_channel_3.gif" alt="Ground truth periodic shear flow simulation showing fluid velocity patterns">
      <img src="assets/nextstep/twophase_gt_channel_2.gif" alt="Ground truth two-phase flow simulation with phase boundaries">
      <img src="assets/nextstep/euler_peri_gt_channel_2.gif" alt="Ground truth Euler equations simulation with periodic boundary conditions">
      <img src="assets/nextstep/rb_gt_channel_2.gif" alt="Ground truth Rayleigh-Bénard convection simulation showing thermal flow patterns">
      <img src="assets/nextstep/shear_pred_channel_3.gif" alt="GPhyT prediction of periodic shear flow simulation">
      <img src="assets/nextstep/twophase_pred_channel_2.gif" alt="GPhyT prediction of two-phase flow simulation">
      <img src="assets/nextstep/euler_peri_pred_channel_2.gif" alt="GPhyT prediction of Euler equations with periodic boundaries">
      <img src="assets/nextstep/rb_pred_channel_2.gif" alt="GPhyT prediction of Rayleigh-Bénard convection simulation">
  </div>

## Generalization to New Physics

The main point of a Physics Foundation Model is to generalize to new physics. Below, we show some examples of GP<sub>hy</sub>T predicting **completely unseen physical systems – no finetuning** (bottom row) against the ground truth (top row). Left and center-left are examples of novel boundary conditions, while center-right and right are examples of completely new physics.
Hopefully, you agree that the new boundary conditions are no challenge for the model. The new physics examples are not perfect, as you can see some artifacts, but still very impressive for a model that has never seen anything like this before.

  <div class="gif-grid-4x2-nospace">
      <img src="assets/nextstep/open_obj_gt_channel_3.gif" alt="Ground truth flow simulation around obstacle with open boundary conditions">
      <img src="assets/nextstep/euler_open_gt_channel_2.gif" alt="Ground truth Euler equations with open boundary conditions">
      <img src="assets/nextstep/supersonic_gt_channel_1.gif" alt="Ground truth supersonic flow simulation showing shock waves">
      <img src="assets/nextstep/turbulent_gt_channel_2.gif" alt="Ground truth turbulent radative layer simulation">
      <img src="assets/nextstep/open_obj_pred_channel_3.gif" alt="GPhyT prediction of flow around obstacle with open boundaries">
      <img src="assets/nextstep/euler_open_pred_channel_2.gif" alt="GPhyT prediction of Euler equations with open boundaries">
      <img src="assets/nextstep/supersonic_pred_channel_1.gif" alt="GPhyT prediction of supersonic flow with shock wave formation">
      <img src="assets/nextstep/turbulent_pred_channel_2.gif" alt="GPhyT prediction of turbulent radiative layer simulation">
  </div>

## Long-Horizon Predictions

Our final experiment are long-horizon predictions, i.e. autoregressive rollouts of the model. This task is incredibly difficult and not current Physics Foundation Model can achieve high enough accuracy to actually replace a numerical solver. Yet, at least for 50 initial time steps, GP<sub>hy</sub>T produces great results for some systems, as shown below. The shear flow (left) and Euler periodic (right) examples are almost indistinguishable from the ground truth. For the highly chaotic Rayleigh-Bénard system (center), the results look reasonable, but the fine details differ quite a bit.

  <div class="gif-grid-3x2-nospace">
      <img src="assets/rollout/sf_gt_vel_mag.gif" alt="Ground truth shear flow velocity magnitude rollout">
      <img src="assets/rollout/rb_gt_density.gif" alt="Ground truth Rayleigh-Bénard density field rollout">
      <img src="assets/rollout/euler_peri_gt_vel_mag.gif" alt="Ground truth Euler periodic velocity magnitude rollout">
      <img src="assets/rollout/sf_pred_vel_mag.gif" alt="GPhyT prediction shear flow velocity magnitude rollout">
      <img src="assets/rollout/rb_pred_density.gif" alt="GPhyT prediction Rayleigh-Bénard density field rollout">
      <img src="assets/rollout/euler_peri_pred_vel_mag.gif" alt="GPhyT prediction Euler periodic velocity magnitude rollout">
  </div>

Long-horizon predictions for unknown physics are still out of reach, as errors accumulate quickly. But we are confident that with more data and better architectures, we will get there eventually.


## What's next?

For a true Physics Foundation Model, we need to solve two practical challenges:

1. How to scale to more physics? We only trained on 6 systems, but the physical world is much more diverse. We need a lot more data!!! Espcially 3D simulations are needed, which are much more expensive to generate. Also, not all physics is best described by pixel data, we also need to include particle-based methods and meshes, which in turn means Graph Neural Networks or similar approaches as encoders. Finally, real-world data could be included, for examples experimental data from fluid dynamics, material science or even astronomy.
2. How to ensure physical consistency? LLMs hallucinations are a known problem, but for physics, this is even more critical. A model that violates conservation of mass or energy is absolutely useless. We need to find ways to enforce physical laws in the model, either through architecture, loss functions or data augmentation.

## Meta-Questions

While building this Physics Foundation Model, some more fundamental questions arose:

1. What is the natural representation of physics? Is it pixel data, particles, meshes, graphs or something else? Our numerical solvers mostly run on mesh and particle data, but these are often also approximations. Should we rather learn from real-world data, for example from videos of the natural world?
2. A world model, i.e. a system that models the real world, might be a better approach than a surrogate model for numerical solvers. Such a world model could perhaps learn a more natural representation of physics, similar to how humans understand the world. 