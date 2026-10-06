"""Each operation's optimizer must step on that operation's own gradients only.

The model is GAN-shaped and wired like a vocoder config: the discriminator
operation runs the generator, detaches its output and scores real and fake; the
generator operation then scores the generator output with the discriminator
under grad, so the generator loss flows through the discriminator's parameters.
"""
import functools

import torch

import hypatorch
from hypatorch import Trainer


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        y = self.lin(x)
        return y


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def forward(self, y, y_hat):
        d_real = self.lin(y)
        d_fake = self.lin(y_hat)
        return d_real, d_fake


class Detach(torch.nn.Module):
    def forward(self, x):
        x_detached = x.detach()
        return x_detached


def discriminator_loss(real, fake):
    return torch.mean((1 - real) ** 2) + torch.mean(fake ** 2)


def generator_loss(fake):
    return torch.mean((1 - fake) ** 2)


def _sgd():
    return functools.partial(torch.optim.SGD, lr=0.1)


def _build_gan():
    torch.manual_seed(0)
    operations = {
        "update_discriminator": {
            "optimizer": _sgd(),
            "optimize_submodules": ["discriminator"],
            "mappings": [
                {"generator": {"inputs": {"x": "z"}, "outputs": {"y": "y_pred"}, "calculate_grad": True}},
                {"detach": {"inputs": {"x": "y_pred"}, "outputs": {"x_detached": "y_pred_detached"}, "calculate_grad": True}},
                {"discriminator": {
                    "inputs": {"y": "y", "y_hat": "y_pred_detached"},
                    "outputs": {"d_real": "d_real", "d_fake": "d_fake"},
                    "calculate_grad": True,
                }},
            ],
            "losses": [
                hypatorch.HypaAssessment(
                    assessment=discriminator_loss,
                    name="discriminator_loss",
                    inputs={"real": "d_real", "fake": "d_fake"},
                ),
            ],
        },
        "update_generator": {
            "optimizer": _sgd(),
            "optimize_submodules": ["generator"],
            "mappings": [
                {"discriminator": {
                    "inputs": {"y": "y", "y_hat": "y_pred"},
                    "outputs": {"d_real": "d_real_update", "d_fake": "d_fake_update"},
                    "calculate_grad": True,
                }},
            ],
            "losses": [
                hypatorch.HypaAssessment(
                    assessment=generator_loss,
                    name="generator_loss",
                    inputs={"fake": "d_fake_update"},
                ),
            ],
        },
    }
    return hypatorch.Model(
        submodules={
            "generator": Generator(),
            "discriminator": Discriminator(),
            "detach": Detach(),
        },
        operations=operations,
    )


def _batch(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "z": torch.randn(8, 4, generator=generator),
        "y": torch.randn(8, 4, generator=generator),
    }


def _clean_discriminator_grads(model, batch):
    params = list(model.discriminator.parameters())
    fake = model.generator(batch["z"]).detach()
    d_real, d_fake = model.discriminator(batch["y"], fake)
    return torch.autograd.grad(discriminator_loss(d_real, d_fake), params)


def _clean_generator_grads(model, batch):
    params = list(model.generator.parameters())
    _, d_fake = model.discriminator(batch["y"], model.generator(batch["z"]))
    return torch.autograd.grad(generator_loss(d_fake), params)


def _record_steps(optimizer, record):
    """Capture the gradients each optimizer step actually applies."""
    step = optimizer.step

    def recording_step(*args, **kwargs):
        record.append([param.grad.clone() for group in optimizer.param_groups for param in group["params"]])
        return step(*args, **kwargs)

    optimizer.step = recording_step


def _setup(grad_accum_steps=1):
    model = _build_gan()
    model.train()
    trainer = Trainer(accelerator="cpu", grad_accum_steps=grad_accum_steps)
    optimizers, schedulers, gradient_clipping = model.configure_optimizers()
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    return model, trainer, optimizers, schedulers, gradient_clipping


def _step(trainer, model, optimizers, schedulers, gradient_clipping, batch, epoch_step):
    trainer.step(
        mode="train",
        model=model,
        input_dict=dict(batch),
        optimizers=optimizers,
        schedulers=schedulers,
        gradient_clipping=gradient_clipping,
        epoch_step=epoch_step,
    )


def test_discriminator_steps_on_its_own_loss_only():
    model, trainer, optimizers, schedulers, gradient_clipping = _setup()
    batch = _batch(0)

    applied = []
    _record_steps(optimizers["update_discriminator"], applied)

    expected = []
    for epoch_step in range(3):
        expected.append(_clean_discriminator_grads(model, batch))
        _step(trainer, model, optimizers, schedulers, gradient_clipping, batch, epoch_step)

        # Nothing the generator operation computed may wait for the next step.
        for param in model.discriminator.parameters():
            assert param.grad is None

    assert len(applied) == 3
    for step_applied, step_expected in zip(applied, expected):
        for got, want in zip(step_applied, step_expected):
            torch.testing.assert_close(got, want)


def test_generator_still_trains_through_the_discriminator():
    model, trainer, optimizers, schedulers, gradient_clipping = _setup()
    batch = _batch(0)

    applied = []
    _record_steps(optimizers["update_generator"], applied)

    for epoch_step in range(3):
        # The discriminator steps first within a step, so score the generator
        # with the discriminator the generator operation will actually see.
        discriminator_step = optimizers["update_discriminator"].step
        expected = None

        def step_then_expect(*args, **kwargs):
            nonlocal expected
            result = discriminator_step(*args, **kwargs)
            expected = _clean_generator_grads(model, batch)
            return result

        optimizers["update_discriminator"].step = step_then_expect
        _step(trainer, model, optimizers, schedulers, gradient_clipping, batch, epoch_step)
        optimizers["update_discriminator"].step = discriminator_step

        for got, want in zip(applied[-1], expected):
            assert torch.count_nonzero(got) > 0
            torch.testing.assert_close(got, want)


def test_accumulation_keeps_only_the_operations_own_gradients():
    model, trainer, optimizers, schedulers, gradient_clipping = _setup(grad_accum_steps=2)
    batches = [_batch(0), _batch(1)]

    # No optimizer steps before the discriminator's, so all of its micro-batch
    # gradients are taken at the initial parameters.
    per_batch = [_clean_discriminator_grads(model, batch) for batch in batches]
    expected = [(first + second) / 2 for first, second in zip(*per_batch)]

    applied = []
    _record_steps(optimizers["update_discriminator"], applied)

    for epoch_step, batch in enumerate(batches):
        _step(trainer, model, optimizers, schedulers, gradient_clipping, batch, epoch_step)

    assert len(applied) == 1
    for got, want in zip(applied[0], expected):
        torch.testing.assert_close(got, want)
