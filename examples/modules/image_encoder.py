import torch

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features,
            out_features,
        )

    def forward(
        self,
        x,
    ):
        x = torch.relu(
            self.linear(
                x.view(
                    x.size(0),
                    -1,
                )
            )
        )

        return x