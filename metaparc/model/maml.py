"""Model Agnostic Meta Learning

Author: Florian Wiesner
Date: 2025-03-07

This module implements the MAML (Model-Agnostic Meta-Learning) algorithm as described in the paper
'Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks' by Finn et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict


class MAML(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        first_order: bool = False,
        num_inner_steps: int = 1,
        learn_inner_lr: bool = False,
    ):
        """Initialize MAML.

        Parameters
        ----------
        base_model : nn.Module
            Base model to be meta-learned
        inner_lr : float, default=0.01
            Learning rate for the inner loop optimization
        meta_lr : float, default=0.001
            Learning rate for the meta-optimization
        first_order : bool, default=False
            If True, use first-order approximation
        num_inner_steps : int, default=1
            Number of gradient steps in the inner loop
        learn_inner_lr : bool, default=False
            If True, learn the inner learning rate
        """
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.first_order = first_order
        self.num_inner_steps = num_inner_steps
        self.learn_inner_lr = learn_inner_lr

        # Initialize meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=meta_lr)

        # If learning inner learning rate, create parameter for each weight
        if learn_inner_lr:
            self.learned_lrs = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.tensor(inner_lr))
                    for name, _ in self.base_model.named_parameters()
                }
            )

    def clone_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of model with same architecture but different parameter values.

        Parameters
        ----------
        model : nn.Module
            Model to be cloned

        Returns
        -------
        nn.Module
            A copy of the model with same architecture but different parameter values
        """
        clone = type(model)()  # Create new instance with same architecture
        clone.load_state_dict(model.state_dict())
        return clone

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        model: Optional[nn.Module] = None,
        create_graph: bool = False,
    ) -> nn.Module:
        """Perform inner loop optimization to adapt model parameters to task.

        Parameters
        ----------
        support_x : torch.Tensor
            Support set inputs
        support_y : torch.Tensor
            Support set labels
        model : nn.Module, optional
            Model to adapt (if None, use self.base_model)
        create_graph : bool, default=False
            Whether to create computation graph (needed for meta-update)

        Returns
        -------
        nn.Module
            Adapted model
        """
        if model is None:
            model = self.base_model

        # Clone model to avoid modifying original parameters during inner loop
        adapted_model = self.clone_model(model)

        for _ in range(self.num_inner_steps):
            support_pred = adapted_model(support_x)
            inner_loss = F.mse_loss(support_pred, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                inner_loss,
                adapted_model.parameters(),
                create_graph=create_graph,
                allow_unused=True,
            )

            # Update parameters
            for (name, param), grad in zip(adapted_model.named_parameters(), grads):
                if grad is not None:
                    lr = (
                        self.learned_lrs[name] if self.learn_inner_lr else self.inner_lr
                    )
                    param.data = param.data - lr * grad

        return adapted_model

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
        """Forward pass for meta-learning.

        Parameters
        ----------
        support_x : torch.Tensor
            Support set inputs
        support_y : torch.Tensor
            Support set labels
        query_x : torch.Tensor
            Query set inputs
        query_y : torch.Tensor
            Query set labels

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, nn.Module]
            Tuple containing:
            - Query set predictions
            - Query set loss
            - Adapted model
        """
        # Adapt model to task using support set
        adapted_model = self.inner_loop(
            support_x, support_y, create_graph=not self.first_order
        )

        # Evaluate on query set
        query_pred = adapted_model(query_x)
        query_loss = F.mse_loss(query_pred, query_y)

        return query_pred, query_loss, adapted_model

    def meta_train_step(
        self, tasks: List[Dict[str, torch.Tensor]]
    ) -> Tuple[float, List[float]]:
        """Perform single meta-training step.

        Parameters
        ----------
        tasks : List[Dict[str, torch.Tensor]]
            List of task dictionaries, each containing:
            - 'support_x': Support set inputs
            - 'support_y': Support set labels
            - 'query_x': Query set inputs
            - 'query_y': Query set labels

        Returns
        -------
        Tuple[float, List[float]]
            Tuple containing:
            - Average meta-loss across tasks
            - List of query losses for each task
        """
        query_losses = []

        self.meta_optimizer.zero_grad()

        for task in tasks:
            # Forward pass
            _, query_loss, _ = self.forward(
                task["support_x"], task["support_y"], task["query_x"], task["query_y"]
            )
            query_losses.append(query_loss.item())

            # Accumulate gradients
            query_loss.backward()

        # Update meta-parameters
        self.meta_optimizer.step()

        return np.mean(query_losses), query_losses

    def adapt_to_task(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_adapt_steps: Optional[int] = None,
    ) -> nn.Module:
        """Adapt model to new task using support set.

        Parameters
        ----------
        support_x : torch.Tensor
            Support set inputs
        support_y : torch.Tensor
            Support set labels
        num_adapt_steps : int, optional
            Number of adaptation steps (if None, use self.num_inner_steps)

        Returns
        -------
        nn.Module
            Adapted model for the new task
        """
        if num_adapt_steps is None:
            num_adapt_steps = self.num_inner_steps

        # Store original number of steps
        original_steps = self.num_inner_steps
        self.num_inner_steps = num_adapt_steps

        # Adapt to task
        adapted_model = self.inner_loop(support_x, support_y, create_graph=False)

        # Restore original number of steps
        self.num_inner_steps = original_steps

        return adapted_model
