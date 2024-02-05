import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_seq_key = 23

    config.model = new_dict(
        n_flow_layers=[
            new_dict(
                type="bijection",
                transformer=new_dict(
                    type="nsf", n_params=8, range_min=-5.0, range_max=5.0
                ),
                conditioner=new_dict(
                    type="mlp",
                    ndim_hidden_layers=512,
                    n_hidden_layers=2,
                    activation="relu",
                ),
            ),
            new_dict(
                type="bijection",
                transformer=new_dict(
                    type="nsf", n_params=8, range_min=-5.0, range_max=5.0
                ),
                conditioner=new_dict(
                    type="mlp",
                    ndim_hidden_layers=512,
                    n_hidden_layers=2,
                    activation="relu",
                ),
            ),
            new_dict(
                type="funnel",
                reduction_factor=0.75,
                decoder=new_dict(
                    type="mlp",
                    ndim_hidden_layers=512,
                    n_hidden_layers=2,
                    n_params=2,
                    activation="relu",
                ),
                transformer=new_dict(
                    type="nsf", n_params=8, range_min=-5.0, range_max=5.0
                ),
                conditioner=new_dict(
                    type="mlp",
                    ndim_hidden_layers=512,
                    n_hidden_layers=2,
                    activation="relu",
                ),
            ),
            new_dict(
                type="bijection",
                transformer=new_dict(
                    type="nsf", n_params=8, range_min=-5.0, range_max=5.0
                ),
                conditioner=new_dict(
                    type="mlp",
                    ndim_hidden_layers=512,
                    n_hidden_layers=2,
                    activation="relu",
                ),
            ),
            new_dict(
                type="bijection",
                transformer=new_dict(
                    type="nsf", n_params=8, range_min=-5.0, range_max=5.0
                ),
                conditioner=new_dict(
                    type="mlp",
                    ndim_hidden_layers=512,
                    n_hidden_layers=2,
                    activation="relu",
                ),
            ),
        ],
    )

    config.training = new_dict(
        n_iter=2000,
        batch_size=64,
        shuffle_data=True,
        train_val_split=0.9,
        rng_seq_key=42,
    )

    config.prediction = new_dict(
        rng_seq_key=42,
        batch_size=64,
    )

    config.optimizer = new_dict(
        name="adamw",
        params=new_dict(
            learning_rate=0.0001,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-8,
        ),
    )

    return config
