# 作者: 晨星
"""eval —— 度量学习评测指标（MAP@R / Recall@k / kNN 精度 / NMI-ARI）。"""

from .metrics import knn_accuracy, map_at_r_score, nmi_ari, recall_at_k

__all__ = ["knn_accuracy", "map_at_r_score", "nmi_ari", "recall_at_k"]
