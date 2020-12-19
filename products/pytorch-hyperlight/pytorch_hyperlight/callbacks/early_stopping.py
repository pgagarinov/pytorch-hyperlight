from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class EarlyStoppingWithGracePeriod(EarlyStopping):

    def __init__(self, *args, grace_period=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__grace_period_count = grace_period

    def _run_early_stopping_check(self, *args, **kwargs):
        if self.__grace_period_count == 0:
            super()._run_early_stopping_check(*args, **kwargs)
        else:
            self.__grace_period_count -= 1

