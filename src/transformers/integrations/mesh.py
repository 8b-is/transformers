import torch
from torch import nn

from ..modeling_utils import PreTrainedModel


class MeshRouterWrapper(PreTrainedModel):
    """
    Top-level wrapper that implements a Network-Level Router (Mesh) over multiple HF models.
    The wrapped models are treated as experts. A top-level gate dispatches tokens to them.
    """

    def __init__(self, config, experts: list[PreTrainedModel]):
        super().__init__(config)
        self.n_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        # Assume the first expert defines the vocabulary size
        vocab_size = experts[0].config.vocab_size

        # A simple token-level routing gate based on an Embedding
        self.gate_embedding = nn.Embedding(vocab_size, self.n_experts)

    def _gate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the routing probabilities for each expert.
        Args:
            input_ids: (batch_size, seq_len)
        Returns:
            gate_probs: (batch_size, seq_len, n_experts)
        """
        logits = self.gate_embedding(input_ids)
        return torch.softmax(logits, dim=-1)

    def forward(self, input_ids=None, **kwargs):
        """
        Forward pass for training or normal inference.
        Note: The generation loop in GenerationMixin intercepts execution
        when `experts` and `_gate` are present.
        """
        gate = self._gate(input_ids)

        expert_outputs = [expert(input_ids=input_ids, **kwargs) for expert in self.experts]
        logits = sum(gate[:, :, i].unsqueeze(-1) * expert_outputs[i].logits for i in range(self.n_experts))

        past_key_values = (
            tuple(e.past_key_values for e in expert_outputs) if hasattr(expert_outputs[0], "past_key_values") else None
        )

        return type(expert_outputs[0])(
            logits=logits,
            past_key_values=past_key_values,
        )
