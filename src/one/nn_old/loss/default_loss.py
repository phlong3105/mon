#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torch import nn

from one.core import LOSSES

# MARK: - Register

LOSSES.register(name="bce_loss", 			    module=nn.BCELoss)
LOSSES.register(name="bce_with_logits_loss",    module=nn.BCEWithLogitsLoss)
LOSSES.register(name="cosine_embedding_loss",   module=nn.CosineEmbeddingLoss)
LOSSES.register(name="cross_entropy_loss",	    module=nn.CrossEntropyLoss)
LOSSES.register(name="ctc_loss",	  		    module=nn.CTCLoss)
LOSSES.register(name="gaussian_nll_loss",       module=nn.GaussianNLLLoss)
LOSSES.register(name="hinge_embedding_loss",    module=nn.HingeEmbeddingLoss)
LOSSES.register(name="kl_div_loss",	  		    module=nn.KLDivLoss)
# LOSSES.register(name="l1_loss",	  		        module=nn.L1Loss)
LOSSES.register(name="margin_ranking_loss",	    module=nn.MarginRankingLoss)
# LOSSES.register(name="mse_loss",	  		    module=nn.MSELoss)
LOSSES.register(name="multi_label_margin_loss", module=nn.MultiLabelMarginLoss)
LOSSES.register(name="multi_margin_loss",       module=nn.MultiMarginLoss)
LOSSES.register(name="nll_loss",   	   		    module=nn.NLLLoss)
LOSSES.register(name="poisson_nll_loss",   	    module=nn.PoissonNLLLoss)
# LOSSES.register(name="smooth_l1_loss",   	    module=nn.SmoothL1Loss)
LOSSES.register(name="soft_margin_loss",   	    module=nn.SoftMarginLoss)
LOSSES.register(name="triplet_margin_loss",     module=nn.TripletMarginLoss)
LOSSES.register(name="triplet_margin_with_distance_Loss", module=nn.TripletMarginWithDistanceLoss)
