import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_key = 23
    config.output_size = 32
    config.num_categories = -1

    embedding_dim = 64
    n_diffusions = 1000
    config.model = new_dict(
        n_diffusions=n_diffusions,

        use_prior_loss=True,
        use_rounding_loss=True,
        use_l0_loss=True,
        stop_gradient_at_logits=True,

        noise_schedule="cosine",
        predict_z0=True,

        embedding=new_dict(
            embedding_dim=embedding_dim,
            normalise_embeddings=True
        ),
        encoder=new_dict(
            positional_encoding_len=1024,
            num_encoder_layers=4,
            num_attention_heads=4,
            qkv_dim=embedding_dim,
            encoder_num_hidden_layers=2,
            encoder_mlp_dim=512,
            encoder_dropout_rate=0.1,
            activation="swish"
        ),
        score_model=new_dict(
            n_diffusions=n_diffusions,
            embedding_dim=embedding_dim,

            time_embedding_dim=256,
            time_embedding_num_hidden_layers=2,
            time_embedding_activation="swish",

            pre_projection_dim=256,
            num_layers_pre_projection_dim=2,

            dropout_rate=0.1,
            activation="swish",

            uses_mask=True,
            prefix_mask_size=int(config.output_size / 4),
            random_suffix_mask_size=int(config.output_size / 4),

            use_classier_free_guidance=False,
            use_self_conditioning=True,
            use_mask_as_input=True,
            project_mask_to_embedding_dim=True
        ),
    )

    config.data = new_dict(
        rng_key=64,
    )

    config.training = new_dict(
        rng_key=23,
        n_iter=100,
        batch_size=64,
        shuffle_data=True,
        train_val_split=0.95,
        early_stopping_patience=10,
        early_stopping_delta=100
    )

    config.simulation = new_dict(
        rng_key=42,
        batch_size=64,
    )

    config.optimizer = new_dict(
        name='adamw',
        params=new_dict(
            learning_rate=0.0003,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-8,
        ),
        warmup=new_dict(
            do_warmup_schedule=False,
            start_learning_rate=0.00001,
        ),
        learning_rate_decay=new_dict(
            do_learning_rate_decay_schedule=True,
            exponential_decay_rate=0.99,
            end_learning_rate=0.00001,
            decay_type="linear"
        ),
        post_decay_schedule=new_dict(
            do_constant_learning_rate_after_schedule=False
        ),
        gradient_transform=new_dict(
            do_gradient_clipping=False,
            gradient_clipping=1.0
        ),
        num_train_steps=10_000,
        num_warmup_steps=100,
    )

    return config
