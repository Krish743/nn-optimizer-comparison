import torch.optim as optim

def get_optimizer(name, model, lr):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)

    elif name == "Momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif name == "NAG":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True
        )

    elif name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)

    elif name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)

    elif name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=lr)
    elif name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
