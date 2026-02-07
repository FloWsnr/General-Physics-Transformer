import torch
import torch.optim as optim


def _get_linear_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_updates: int,
    start_factor: float,
    end_factor: float,
) -> optim.lr_scheduler.LinearLR:
    """Create a linear learning rate scheduler."""
    return optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=num_updates,
    )


def _get_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_updates: int,
    min_lr: float,
) -> optim.lr_scheduler.CosineAnnealingLR:
    """Create a cosine learning rate scheduler."""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=min_lr
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lrs_config: dict,
    total_batches: int,
    total_batches_trained: int = 0,
) -> optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler.
    Options are only linear warmup or linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    lrs_config : dict
        Learning rate scheduler configuration
    total_batches : int
        Total number of batches for training
    total_batches_trained : int
        Total number of batches trained so far
    Returns
    -------
    optim.lr_scheduler.SequentialLR
        Learning rate scheduler
    """
    learning_rate = optimizer.param_groups[0]["lr"]
    total_updates_remaining = total_batches - total_batches_trained

    first_stage = lrs_config["first_stage"]
    stages = [first_stage]
    if "second_stage" in lrs_config:
        second_stage = lrs_config["second_stage"]
        stages.append(second_stage)
    if "third_stage" in lrs_config:
        third_stage = lrs_config["third_stage"]
        stages.append(third_stage)

    ############################################################
    ###### Get batches for each stage #########################
    ############################################################
    milestones = []
    for stage in stages:
        stage_updates = stage["num_updates"]
        milestones.append(stage_updates)

    # check if one milestone is -1, but only one
    if sum(1 for x in milestones if x == -1) > 1:
        raise ValueError("Only one milestone can be -1")

    if -1 in milestones:
        # compute actual number of updates for this stage
        # this is the number of updates for all stages except this one
        updates = 0
        for i in range(len(milestones)):
            if milestones[i] != -1:
                updates += milestones[i]
        # this is the number of updates for this stage
        milestones[milestones.index(-1)] = total_updates_remaining - updates

    schedulers = []
    ############################################################
    ############################################################
    ###### First stage #########################################
    ############################################################
    for milestone, stage in zip(milestones, stages):
        stage_name = stage["name"]
        if stage_name == "LinearLR":
            start_factor = float(stage["start_factor"])
            end_factor = float(stage["end_factor"])
            scheduler = _get_linear_lr_scheduler(
                optimizer,
                num_updates=milestone,
                start_factor=start_factor,
                end_factor=end_factor,
            )
        elif stage_name == "CosineAnnealingLR":
            min_lr = float(stage["end_factor"]) * learning_rate
            scheduler = _get_cosine_lr_scheduler(
                optimizer,
                num_updates=milestone,
                min_lr=min_lr,
            )
        schedulers.append(scheduler)

    # remove the last milestone, bc seqLR needs 1 less than num of schedulers
    milestones = milestones[:-1]
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )

    return scheduler
