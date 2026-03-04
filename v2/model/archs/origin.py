from ._origin_net import General_VolFormer


def infer_n_colors(dataset_name: str, n_colors: int):
    if n_colors > 0:
        return n_colors
    mapping = {
        "Cave": 31,
        "Chikusei": 128,
        "Pavia": 102,
    }
    return mapping.get(dataset_name, 121)


def build_model(options, device, gpu_count):
    model_cfg = options.model

    colors = infer_n_colors(model_cfg.dataset_name, model_cfg.n_colors)
    model = General_VolFormer(
        n_subs=model_cfg.n_subs,
        n_ovls=model_cfg.n_ovls,
        n_colors=colors,
        n_blocks=model_cfg.n_blocks,
        n_feats=model_cfg.n_feats,
        n_scale=model_cfg.n_scale,
        res_scale=0.1,
        use_share=True,
        vf_embed_dim=model_cfg.vf_embed_dim,
        vf_depth=model_cfg.vf_depth,
        vf_layers=model_cfg.vf_layers,
        vf_num_heads=model_cfg.vf_num_heads,
        vf_use_checkpoint=bool(model_cfg.vf_use_checkpoint),
    )
    model.to(device)
    return model, colors
