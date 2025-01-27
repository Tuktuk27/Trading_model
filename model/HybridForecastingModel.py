import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridForecastingModel(nn.Module):
    def __init__(self, input_shapes, hidden_sizes, num_heads, fusion_size, output_sizes, strength = False):
        """
        Args:
            input_shapes (dict): Input shapes for each dataset (key: dataset name, value: (seq_len, num_entities, num_features)).
            hidden_sizes (dict): Hidden sizes for each dataset's processing branch.
            num_heads (int): Number of heads for attention layers.
            fusion_size (int): Size of the fusion layer output.
            output_sizes (dict): Output sizes for each task (key: dataset name, value: number of outputs).
        """
        super(HybridForecastingModel, self).__init__()
        
        self.branches = nn.ModuleDict()
        self.entity_embeddings = nn.ModuleDict()
        self.metric_embeddings = nn.ModuleDict()
        self.tot_per_type = []
        self.strength = strength

        self.linear_projection = nn.ModuleDict()

        # Entity-specific branches
        for dataset, (seq_len, num_entities, num_features) in input_shapes.items():
            self.entity_embeddings[dataset] = nn.Embedding(num_entities, hidden_sizes[dataset])
            self.metric_embeddings[dataset] = nn.Embedding(num_features, hidden_sizes[dataset])


            self.branches[dataset] = nn.Sequential(
                nn.Linear(num_entities*num_features*hidden_sizes[dataset], hidden_sizes[dataset]),  # Add input projection to match LSTM input size --> Reduce large feature size to 128 # 
                nn.LSTM(hidden_sizes[dataset], hidden_sizes[dataset], batch_first=True),
                nn.Linear(hidden_sizes[dataset], hidden_sizes[dataset] // 2)
            )

            projection = hidden_sizes[dataset] #*10  ## Increase the projection of transformer to work on

            self.tot_per_type.append(projection)

            self.linear_projection[dataset] = nn.Linear(num_entities * num_features * hidden_sizes[dataset], projection)
        
        # Cross-domain interaction (attention-based)
        total_hidden = sum(self.tot_per_type)
        # total_hidden = sum(hidden_sizes.values())

        self.cross_attention = nn.MultiheadAttention(embed_dim=total_hidden, num_heads=num_heads, batch_first=True)
        self.cross_fc = nn.Sequential(
            nn.Linear(total_hidden, total_hidden // 2),
            nn.SiLU()
        )

        # # Fusion layer
        fusion_input_size = total_hidden // 2 + sum([hidden_sizes[d] // 2 for d in input_shapes])

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_size),
            nn.SiLU()
        )

        # # Output heads
        # self.output_heads = nn.ModuleDict({
        #     dataset: nn.ModuleDict({
        #         "direction": nn.Linear(fusion_size, output_sizes[dataset]),
        #         "strength": nn.Linear(fusion_size, output_sizes[dataset]),
        #         "confidence": nn.Linear(fusion_size, output_sizes[dataset])
        #     }) for dataset in output_sizes
        # })


        # Add a gating mechanism (optional)
        self.gating_layer = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_size),  # Linear transformation for gating
            nn.Sigmoid()  # Sigmoid to output gate values between 0 and 1
        )
        print(f"{fusion_size = }")
        print(f"{num_heads = }")
        # Attention layer (optional)
        self.attention_layer = nn.MultiheadAttention(embed_dim=fusion_size, num_heads=num_heads, batch_first=True)

        # Shared intermediate layer
        self.shared_output_layer = nn.Sequential(
            nn.Linear(fusion_size, fusion_size // 2),
            nn.SiLU(),
            nn.Linear(fusion_size // 2, fusion_size // 4)  # Shared representation
        )

        if self.strength:
            # Task-specific layers
            self.output_heads = nn.ModuleDict({
                dataset: nn.ModuleDict({
                    "direction": nn.Sequential(
                        nn.Linear(fusion_size // 4, output_sizes[dataset]),  # Binary classification
                        nn.Sigmoid()  # Output in [0, 1]
                    ),
                    "strength": nn.Sequential(
                        nn.Linear(fusion_size // 4, output_sizes[dataset]),  # Regression output (no activation)
                    ),
                    "confidence": nn.Sequential(
                        nn.Linear(fusion_size // 4, output_sizes[dataset]),  # Confidence
                        nn.Softplus()  # Non-negative confidence
                    )
                }) for dataset in output_sizes
            })
        else:
            # Task-specific layers
            self.output_heads = nn.ModuleDict({
                dataset: nn.ModuleDict({
                    "direction": nn.Sequential(
                        nn.Linear(fusion_size // 4, output_sizes[dataset]),  # Binary classification
                        nn.Sigmoid()  # Output in [0, 1]
                    )
                }) for dataset in output_sizes
            })

    def forward(self, data):
        """
        Args:
            data (dict): Input tensors for each dataset (key: dataset name, value: tensor).
                         Each tensor has shape (batch_size, seq_len, num_entities, num_features).

        Returns:
            dict: Predictions for each dataset.
                  Format: {dataset_name: {"direction": tensor, "strength": tensor, "confidence": tensor}}.
        """
        branch_outputs = []
        cross_inputs = []
        attention_masks = []

        # Apply masking to handle NaN values in the data
        data = self.apply_mask(data)  # Apply mask to input data (this ignores NaNs)

        max_seq_len = max(x.shape[1] for x in data.values())  # Determine the maximum sequence length across datasets

        print(f"{data = }")
        print(f"{data["forex"].shape = }")
        print(f"{data["commodities"].shape = }")

        for dataset, x in data.items():

            batch_size, seq_len, num_entities, num_features = x.shape

            # # Entity and metric embeddings
            entity_emb = self.entity_embeddings[dataset](torch.arange(num_entities, device=x.device)).unsqueeze(0).unsqueeze(0)
            metric_emb = self.metric_embeddings[dataset](torch.arange(num_features, device=x.device)).unsqueeze(0).unsqueeze(0)


            # Reshape and repeat embeddings to match dimensions
            entity_emb = entity_emb.repeat(batch_size, seq_len, 1, 1)  # (batch, seq_len, num_entities, hidden_size)
            metric_emb = metric_emb.repeat(batch_size, seq_len, 1, 1)  # (batch, seq_len, num_features, hidden_size)


            # Expand and align dimensions for broadcasting
            x_entity_emb = x.unsqueeze(-1) * entity_emb.unsqueeze(3)  # Shape: (batch, seq_len, num_entities, num_features, hidden_size)

            x_emb = x_entity_emb + metric_emb.unsqueeze(2)           # Shape: (batch, seq_len, num_entities, num_features, hidden_size)


            # Ensure the data is of type float32 before passing to LSTM
            x_emb = x_emb.float()  # This casts the tensor to float32

            # Flatten the last two dimensions for processing
            x_emb = x_emb.view(batch_size, seq_len, -1)  # Shape: (batch, seq_len, num_entities * num_features * hidden_size)


            branch_out = self.branches[dataset][0](x_emb)  # Get linearized output

            branch_out, _ = self.branches[dataset][1](branch_out)  # Get LSTM output


            branch_out = branch_out[:, -1, :]  # Take the last time step (batch_size, hidden_size)

            branch_out = self.branches[dataset][2](branch_out)  # Pass through the Linear layer


            # Entity-specific branch
            branch_outputs.append(branch_out)  # No need to index further


            x_emb = self.linear_projection[dataset](x_emb)


            # Pad sequences to the maximum length and create attention masks
            padding = max_seq_len - seq_len
            if padding > 0:
                # Pad along the sequence dimension (2nd dimension in x_emb)
                x_emb = F.pad(x_emb, (0, 0, padding, 0))  
                # x_emb shape remains: (batch_size, max_seq_len, num_entities * num_features * hidden_size)


            mask = torch.ones(batch_size, max_seq_len, x_emb.shape[-1],dtype=torch.bool, device=x.device)

            mask[:, padding:, :] = False  # False for valid positions, True for padded positions
            attention_masks.append(mask)

            # Add the padded x_emb for cross-domain input
            cross_inputs.append(x_emb)


        # Concatenate inputs and masks
        cross_inputs = torch.cat(cross_inputs, dim=-1)  # Concatenate along the feature dimension

        attention_mask = torch.cat(attention_masks, dim=-1)  # Combine all masks


        # Cross-domain interaction with attention
        cross_out, _ = self.cross_attention(cross_inputs, cross_inputs, cross_inputs) #, key_padding_mask=attention_mask)
        cross_out = self.cross_fc(cross_out.mean(dim=1))  # Mean pooling over sequence

        # # Cross-domain interaction
        # cross_inputs = torch.cat(cross_inputs, dim=-1)  # Concatenate across datasets

        # print(f"cross_inputs shape: {cross_inputs.shape}")

        # cross_out, _ = self.cross_attention(cross_inputs, cross_inputs, cross_inputs)
        # cross_out = self.cross_fc(cross_out.mean(dim=1))  # Mean pooling over sequence

        # Fusion
        cat_features = torch.cat(branch_outputs + [cross_out], dim=-1)

        fused_features = self.fusion_layer(cat_features)

        # Apply the gating mechanism
        gated_output = fused_features * self.gating_layer(cat_features)  # Element-wise multiplication

        # Optionally apply attention mechanism (depends on your choice to use it)
        attended_output, _ = self.attention_layer(gated_output, gated_output, gated_output)

        # Pass through shared output layer
        shared_output = self.shared_output_layer(attended_output)

        # Output predictions
        predictions = {}
        if self.strength:
            for dataset in data.keys():
                if dataset in self.output_heads.keys():
                    predictions[dataset] = {
                        "direction": torch.sigmoid(self.output_heads[dataset]["direction"](shared_output)),
                        "strength": self.output_heads[dataset]["strength"](shared_output),
                        "confidence": torch.sigmoid(self.output_heads[dataset]["confidence"](shared_output))
                    }
        else:
            for dataset in data.keys():
                if dataset in self.output_heads.keys():
                    predictions[dataset] = {
                        "direction": torch.sigmoid(self.output_heads[dataset]["direction"](shared_output)),
                    }

        return predictions
    
    def apply_mask(self, x):
        data = {}
        # Create a mask where 1 means the value is valid, and 0 means it's NaN
        for key, value in x.items():
            mask = torch.isnan(value)  # Invert to make 1 for valid values and 0 for NaN   
            masked_value = value.clone()         
            data[key] = masked_value.masked_fill(mask, 0)
        return data

