import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchForecastingModel(nn.Module):
    def __init__(self, input_shapes, global_hidden_size, fusion_hidden_size, num_heads=4):
        super(DualBranchForecastingModel, self).__init__()

        self.entity_branches = nn.ModuleDict()
        self.entity_embeddings = nn.ModuleDict()
        self.metric_embeddings = nn.ModuleDict()
        branch_hidden_sizes = []

        # Create domain-specific branches
        for dataset_name, input_shape in input_shapes.items():
            seq_len, num_entities, num_metrics = input_shape[1:]

            self.entity_embeddings[dataset_name] = nn.Embedding(num_entities, global_hidden_size)
            self.metric_embeddings[dataset_name] = nn.Embedding(num_metrics, global_hidden_size)

            self.entity_branches[dataset_name] = nn.Sequential(
                nn.LSTM(global_hidden_size, global_hidden_size, batch_first=True),
                nn.Linear(global_hidden_size, global_hidden_size // 2)
            )
            branch_hidden_sizes.append(global_hidden_size // 2)

        # Combined data branch
        combined_input_size = sum([shape[-1] * shape[2] for shape in input_shapes.values()])
        self.combined_branch = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=combined_input_size,
                nhead=num_heads,
                dim_feedforward=global_hidden_size * 2,
                activation="gelu",
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(sum(branch_hidden_sizes) + global_hidden_size, fusion_hidden_size),
            nn.ReLU()
        )

        # Output heads
        self.output_heads = nn.ModuleDict({
            "forex": nn.ModuleDict({
                "direction": nn.Linear(fusion_hidden_size, input_shapes['forex'][2]),
                "strength": nn.Linear(fusion_hidden_size, input_shapes['forex'][2]),
                "confidence": nn.Linear(fusion_hidden_size, input_shapes['forex'][2])
            }),
            "commodities": nn.ModuleDict({
                "direction": nn.Linear(fusion_hidden_size, input_shapes['commodities'][2]),
                "strength": nn.Linear(fusion_hidden_size, input_shapes['commodities'][2]),
                "confidence": nn.Linear(fusion_hidden_size, input_shapes['commodities'][2])
            })
        })

    def forward(self, data_dict):
        # Domain-specific branch processing
        branch_outputs = []

        for dataset_name, data in data_dict.items():
            batch_size, seq_len, num_entities, num_metrics = data.shape

            # Embedding step
            entity_emb = self.entity_embeddings[dataset_name](
                torch.arange(num_entities, device=data.device)
            ).unsqueeze(0).unsqueeze(0)
            metric_emb = self.metric_embeddings[dataset_name](
                torch.arange(num_metrics, device=data.device)
            ).unsqueeze(0).unsqueeze(0)

            data_emb = data.unsqueeze(-1) * entity_emb + metric_emb
            data_emb = data_emb.view(batch_size, seq_len, -1)

            branch_out, _ = self.entity_branches[dataset_name](data_emb)
            branch_outputs.append(branch_out[:, -1, :])  # Take the last step

        # Combined data branch
        combined_data = torch.cat([data.view(data.shape[0], data.shape[1], -1) for data in data_dict.values()], dim=-1)
        combined_output = self.combined_branch(combined_data)[:, -1, :]  # Last step of the combined branch

        # Fusion
        fused_features = torch.cat(branch_outputs + [combined_output], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        # Outputs
        predictions = {}
        for dataset_name in self.output_heads.keys():
            predictions[dataset_name] = {
                "direction": torch.sigmoid(self.output_heads[dataset_name]["direction"](fused_features)),
                "strength": self.output_heads[dataset_name]["strength"](fused_features),
                "confidence": self.output_heads[dataset_name]["confidence"](fused_features)
            }

        return predictions