def _early_stopping_enabled(config):
    return bool(config and config.get("enabled", False))


def _make_early_stopping_callback(config, fallback_monitor, callback_cls=None):
    if callback_cls is None:
        from lightning.pytorch.callbacks import EarlyStopping

        callback_cls = EarlyStopping

    config = config or {}
    monitor = config.get("monitor") or fallback_monitor
    patience = int(config.get("patience", 5))
    mode = config.get("mode", "min")
    min_delta = float(config.get("min_delta", 0.0))
    verbose = bool(config.get("verbose", True))

    return callback_cls(
        monitor=monitor,
        patience=patience,
        mode=mode,
        min_delta=min_delta,
        verbose=verbose,
    )
