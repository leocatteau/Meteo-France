from utils.metrics_base import MaskedMetric
from torch.nn import functional as F
import torch

epsilon = 1e-8


class MaskedMAE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMAE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMSE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMSE, self).__init__(metric_fn=F.mse_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=True,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)