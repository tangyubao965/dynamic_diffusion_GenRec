"""Stage-wise sem-id evolution in DDGR (Section 3.6).

The same conditional denoising model is reused in two modes:

1. item-conditioned inference regenerates the sem-id space;
2. user-conditioned training updates the recommender on the regenerated targets.

Recommendation gradients are not propagated through discrete decoding. They
reach the next regeneration round through the shared denoising parameters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Hashable, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from models.diffusion import DiffusionScheduler, DiffusionTrainer


@dataclass
class EvolutionConfig:
    rounds: int = 3
    sem_id_length: int = 10
    inference_steps: int = 10
    fine_tune_epochs_per_round: int = 1
    lambda_set: float = 1.0
    conflict_topk: int = 20
    item_id_key: str = "item_id"
    item_condition_key: str = "item_condition_ids"
    item_attention_mask_key: str = "item_attention_mask"
    user_condition_key: str = "history_ids"
    user_attention_mask_key: str = "history_attention_mask"
    target_item_key: str = "target_item_id"
    target_sem_id_key: str = "target_ids"
    pad_token_id: Optional[int] = None


@dataclass
class GeneratedIdentifier:
    item_id: Hashable
    token_ids: list[int]
    token_log_probs: list[float]
    topk_token_ids: list[list[int]]
    topk_log_probs: list[list[float]]

    @property
    def confidence(self) -> float:
        return float(sum(self.token_log_probs))


@dataclass
class EvolutionRoundStats:
    round_index: int
    num_items: int
    duplicate_items_before_resolution: int
    duplicate_rate_before_resolution: float
    average_total_loss: float
    average_diffusion_loss: float
    average_set_loss: float


def _as_python_ids(value) -> list[Hashable]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, int)):
        return [value]
    return list(value)


def _duplicate_item_count(identifiers: Iterable[GeneratedIdentifier]) -> int:
    groups: dict[tuple[int, ...], int] = {}
    for candidate in identifiers:
        key = tuple(candidate.token_ids)
        groups[key] = groups.get(key, 0) + 1
    return sum(count for count in groups.values() if count > 1)


def resolve_duplicate_sem_ids(
    identifiers: Iterable[GeneratedIdentifier],
) -> dict[Hashable, list[int]]:
    """Resolve duplicate full sem-ids with confidence-based token replacement.

    Higher-confidence assignments are kept first. For a conflicting item, the
    lowest-confidence token position is replaced by its highest-scoring
    alternative that yields an unused full sem-id, matching Appendix B.
    """
    ordered = sorted(identifiers, key=lambda item: item.confidence, reverse=True)
    used: set[tuple[int, ...]] = set()
    resolved: dict[Hashable, list[int]] = {}

    for candidate in ordered:
        current = list(candidate.token_ids)
        current_key = tuple(current)
        if current_key not in used:
            used.add(current_key)
            resolved[candidate.item_id] = current
            continue

        position_order = sorted(
            range(len(current)), key=lambda pos: candidate.token_log_probs[pos]
        )
        replacement = None
        for position in position_order:
            for alternative in candidate.topk_token_ids[position]:
                if int(alternative) == current[position]:
                    continue
                proposal = list(current)
                proposal[position] = int(alternative)
                proposal_key = tuple(proposal)
                if proposal_key not in used:
                    replacement = proposal
                    break
            if replacement is not None:
                break

        if replacement is None:
            raise RuntimeError(
                "unable to resolve a duplicate sem-id from stored alternatives; "
                "increase EvolutionConfig.conflict_topk"
            )
        used.add(tuple(replacement))
        resolved[candidate.item_id] = replacement

    return resolved


class SemIDEvolutionEngine:
    """Run the stage-wise evolution loop using one shared denoising model."""

    def __init__(
        self,
        model,
        scheduler: DiffusionScheduler,
        optimizer: torch.optim.Optimizer,
        config: EvolutionConfig,
        *,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.scheduler = scheduler.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.diffusion = DiffusionTrainer(model, scheduler)

    @torch.no_grad()
    def regenerate_sem_ids(
        self, item_loader: Iterable[Mapping[str, object]]
    ) -> tuple[dict[Hashable, list[int]], dict[str, float]]:
        """Regenerate identifiers under item-side conditions in inference mode."""
        self.model.eval()
        generated: list[GeneratedIdentifier] = []

        for batch in item_loader:
            if self.config.item_id_key not in batch:
                raise KeyError(f"missing batch key: {self.config.item_id_key}")
            if self.config.item_condition_key not in batch:
                raise KeyError(f"missing batch key: {self.config.item_condition_key}")

            item_ids = _as_python_ids(batch[self.config.item_id_key])
            condition_ids = torch.as_tensor(
                batch[self.config.item_condition_key], device=self.device
            ).long()
            attention_mask = batch.get(self.config.item_attention_mask_key)
            if attention_mask is not None:
                attention_mask = torch.as_tensor(
                    attention_mask, device=self.device
                ).bool()

            token_ids, logits = self.diffusion.sample_with_logits(
                condition_ids,
                self.config.sem_id_length,
                num_inference_steps=self.config.inference_steps,
                condition_attention_mask=attention_mask,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs, dim=-1, index=token_ids.unsqueeze(-1)
            ).squeeze(-1)
            topk = min(self.config.conflict_topk, logits.size(-1))
            topk_log_probs, topk_token_ids = torch.topk(
                log_probs, k=topk, dim=-1
            )

            if len(item_ids) != token_ids.size(0):
                raise ValueError("item_id count does not match item condition batch size")
            for row, item_id in enumerate(item_ids):
                generated.append(
                    GeneratedIdentifier(
                        item_id=item_id,
                        token_ids=token_ids[row].detach().cpu().tolist(),
                        token_log_probs=selected_log_probs[row]
                        .detach()
                        .cpu()
                        .tolist(),
                        topk_token_ids=topk_token_ids[row]
                        .detach()
                        .cpu()
                        .tolist(),
                        topk_log_probs=topk_log_probs[row]
                        .detach()
                        .cpu()
                        .tolist(),
                    )
                )

        duplicate_items = _duplicate_item_count(generated)
        sem_id_map = resolve_duplicate_sem_ids(generated)
        num_items = len(generated)
        diagnostics = {
            "num_items": float(num_items),
            "duplicate_items_before_resolution": float(duplicate_items),
            "duplicate_rate_before_resolution": (
                duplicate_items / num_items if num_items else 0.0
            ),
        }
        return sem_id_map, diagnostics

    def _targets_from_batch(
        self,
        batch: Mapping[str, object],
        sem_id_map: Mapping[Hashable, list[int]],
    ) -> Tensor:
        if self.config.target_sem_id_key in batch:
            return torch.as_tensor(
                batch[self.config.target_sem_id_key], device=self.device
            ).long()
        if self.config.target_item_key not in batch:
            raise KeyError(
                f"batch must contain {self.config.target_sem_id_key!r} or "
                f"{self.config.target_item_key!r}"
            )
        item_ids = _as_python_ids(batch[self.config.target_item_key])
        try:
            targets = [sem_id_map[item_id] for item_id in item_ids]
        except KeyError as error:
            raise KeyError(f"target item {error.args[0]!r} has no regenerated sem-id") from error
        return torch.tensor(targets, device=self.device, dtype=torch.long)

    def fine_tune_recommender(
        self,
        recommendation_loader: Iterable[Mapping[str, object]],
        sem_id_map: Mapping[Hashable, list[int]],
    ) -> dict[str, float]:
        """Fine-tune the shared model under user-side conditions."""
        self.model.train()
        totals = {"total_loss": 0.0, "diffusion_loss": 0.0, "set_loss": 0.0}
        steps = 0

        for _ in range(self.config.fine_tune_epochs_per_round):
            for batch in recommendation_loader:
                if self.config.user_condition_key not in batch:
                    raise KeyError(f"missing batch key: {self.config.user_condition_key}")
                history_ids = torch.as_tensor(
                    batch[self.config.user_condition_key], device=self.device
                ).long()
                history_mask = batch.get(self.config.user_attention_mask_key)
                if history_mask is not None:
                    history_mask = torch.as_tensor(
                        history_mask, device=self.device
                    ).bool()
                target_ids = self._targets_from_batch(batch, sem_id_map)

                losses = self.diffusion.compute_losses(
                    target_ids,
                    history_ids,
                    target_set_ids=target_ids,
                    condition_attention_mask=history_mask,
                    lambda_set=self.config.lambda_set,
                    ignore_index=self.config.pad_token_id,
                )
                self.optimizer.zero_grad(set_to_none=True)
                losses["total_loss"].backward()
                self.optimizer.step()

                for key in totals:
                    totals[key] += float(losses[key].detach().cpu())
                steps += 1

        if steps == 0:
            raise ValueError("recommendation_loader produced no batches")
        return {key: value / steps for key, value in totals.items()}

    def run(
        self,
        item_loader: Iterable[Mapping[str, object]],
        recommendation_loader_factory: Callable[
            [Mapping[Hashable, list[int]], int], Iterable[Mapping[str, object]]
        ],
        *,
        output_dir: Optional[str | Path] = None,
        round_callback: Optional[
            Callable[[int, Mapping[Hashable, list[int]], EvolutionRoundStats], None]
        ] = None,
    ) -> tuple[dict[Hashable, list[int]], list[EvolutionRoundStats]]:
        """Alternate sem-id regeneration and recommendation fine-tuning."""
        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        current_sem_ids: dict[Hashable, list[int]] = {}
        all_stats: list[EvolutionRoundStats] = []

        for round_index in range(1, self.config.rounds + 1):
            current_sem_ids, generation_stats = self.regenerate_sem_ids(item_loader)
            recommendation_loader = recommendation_loader_factory(
                current_sem_ids, round_index
            )
            training_stats = self.fine_tune_recommender(
                recommendation_loader, current_sem_ids
            )
            stats = EvolutionRoundStats(
                round_index=round_index,
                num_items=int(generation_stats["num_items"]),
                duplicate_items_before_resolution=int(
                    generation_stats["duplicate_items_before_resolution"]
                ),
                duplicate_rate_before_resolution=float(
                    generation_stats["duplicate_rate_before_resolution"]
                ),
                average_total_loss=training_stats["total_loss"],
                average_diffusion_loss=training_stats["diffusion_loss"],
                average_set_loss=training_stats["set_loss"],
            )
            all_stats.append(stats)

            if output_path is not None:
                with (output_path / f"sem_ids_round_{round_index}.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(
                        {str(key): value for key, value in current_sem_ids.items()},
                        file,
                        indent=2,
                    )
                with (output_path / f"stats_round_{round_index}.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(asdict(stats), file, indent=2)

            if round_callback is not None:
                round_callback(round_index, current_sem_ids, stats)

        return current_sem_ids, all_stats


def run_evolution(
    model,
    scheduler: DiffusionScheduler,
    optimizer: torch.optim.Optimizer,
    item_loader: Iterable[Mapping[str, object]],
    recommendation_loader_factory: Callable[
        [Mapping[Hashable, list[int]], int], Iterable[Mapping[str, object]]
    ],
    config: EvolutionConfig,
    *,
    device: str | torch.device,
    output_dir: Optional[str | Path] = None,
):
    """Convenience function corresponding to Algorithm 1 after initial training."""
    engine = SemIDEvolutionEngine(
        model, scheduler, optimizer, config, device=device
    )
    return engine.run(
        item_loader,
        recommendation_loader_factory,
        output_dir=output_dir,
    )
