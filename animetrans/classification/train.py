from typing import Optional


def train(
        workdir: str,
        dataset_repo_id: str,
        timm_model_name: str,
        num_workers: int = 16,
        max_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-3,
        key_metric: str = 'macro_f1',
        seed: Optional[int] = 0,
        eval_epoch: int = 1,
        eval_threshold: float = 0.4,
        model_args: Optional[dict] = None,
        pretrained_cfg: Optional[dict] = None,

        noise_level: int = 2,
        rotation_ratio: float = 0.0,
        mixup_alpha: float = 0.6,
        cutout_max_pct: float = 0.0,
        cutout_patches: int = 0,
        random_resize_method: bool = True,
        pre_align: bool = True,
        align_size: int = 512,
        tag_categories: Optional[Sequence[int]] = None,
        seen_tag_keys: Optional[List[str]] = None,
        image_key: str = 'webp',
        use_pretrained_weight: bool = True,
        adam_betas: Optional[Tuple[float, float]] = None,
        use_normalize: bool = False,
):
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    if seed is None:
        seed = random.randint(0, (1 << 31) - 1)
    blist = [seed]
    broadcast_object_list(blist, from_process=0)
    seed = blist[0] + accelerator.process_index
    # native random, numpy, torch and faker's seeds are includes
    # if you need to register more library for seeding, see:
    # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
    logging.info(f'Globally set the random seed {seed!r} in process #{accelerator.process_index}.')
    global_seed(seed)
