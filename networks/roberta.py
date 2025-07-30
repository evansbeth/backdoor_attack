from transformers import RobertaModel, RobertaConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch.nn as nn

class CustomRobertaQA(nn.Module):
    def __init__(self, linear_cls, config_name="roberta-base"):
        super().__init__()
        # 1) load config + backbone
        self.config  = RobertaConfig.from_pretrained(config_name)
        self.roberta = RobertaModel.from_pretrained(config_name, config=self.config)

        # 2) replace every nn.Linear in the backbone
        self._replace_linear(self.roberta, linear_cls)

        # 3) QA head: map hidden_size→2 (start, end)
        hidden_size     = self.config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                **kwargs):
        # 1) get base outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # 2) project to (batch, seq_len, 2) then split
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits   = end_logits.squeeze(-1)

        # 3) wrap in the HF QuestionAnsweringModelOutput
        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            # all the rest just passthrough so nobody breaks:
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _replace_linear(self, module, linear_cls):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # swap it out
                new_layer = linear_cls(child.in_features, child.out_features)
                # copy weights/bias
                new_layer.load_state_dict(child.state_dict(), strict=False)
                setattr(module, name, new_layer)
            else:
                # recurse
                self._replace_linear(child, linear_cls)