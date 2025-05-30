import torch
import torch.nn as nn

class AttentionMechanism(nn.Module):
    def __init__(self, latent_dim):
        super(AttentionMechanism, self).__init__()
        self.query_proj = nn.Linear(latent_dim, 64)
        self.key_proj = nn.Linear(latent_dim, 64)
        self.temperature = 8.0  # Scaling factor for sharper attention

    def forward(self, query, keys):
        """
        Apply attention mechanism

        Args:
            query: Test input latent [batch_size, latent_dim]
            keys: Training input latents [num_examples, latent_dim]

        Returns:
            Attention weights [batch_size, num_examples]
        """
        # Project query and keys to a smaller dimension
        q = self.query_proj(query)  # [batch_size, 64]
        k = self.key_proj(keys)     # [num_examples, 64]

        # Compute attention scores
        scores = torch.matmul(q, k.t()) / self.temperature  # [batch_size, num_examples]

        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=1)

        return attention

class AttentionSolver(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super(AttentionSolver, self).__init__()
        self.attention = AttentionMechanism(latent_dim)

        self.transformation_inference = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Output a transformation vector
        )
        self.prediction_module = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, test_latent, train_input_latents, train_output_latents):
        # 1. Infer a transformation from the training pairs
        # For simplicity, let's use the first train pair to infer the transformation
        if train_input_latents.size(0) > 0:
            train_pair = torch.cat([train_input_latents[0].unsqueeze(0), train_output_latents[0].unsqueeze(0)], dim=-1)
            transformation = self.transformation_inference(train_pair)
        else:
            transformation = torch.zeros_like(test_latent.unsqueeze(0)) # Handle cases with no train pairs

        # 2. Apply the inferred transformation to the test latent
        transformed_test_latent = test_latent + transformation.squeeze(0)

        # 3. Combine the transformed test latent with the original test latent
        combined = torch.cat([test_latent, transformed_test_latent], dim=-1)
        predicted_latent = self.prediction_module(combined)

        # You can still incorporate attention if needed
        attention_weights = self.attention(test_latent, train_input_latents)
        weighted_output = torch.matmul(attention_weights, train_output_latents)
        predicted_latent = predicted_latent + weighted_output # Example of incorporating attention

        return predicted_latent, attention_weights