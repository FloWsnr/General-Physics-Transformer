from neuralop.models import FNO


class FourierNeuralOperator(FNO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)
