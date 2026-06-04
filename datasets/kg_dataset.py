"""
datasets/kg_dataset.py — v10
KGDataset cho KGAT repo format.

THAY ĐỔI QUAN TRỌNG v10 so với v7:
  - Nguồn dữ liệu: KGAT repo DUY NHẤT (unified/)
  - File KG: kg_final.txt (không phải kg_full.txt)
  - BẤT BIẾN BẮT BUỘC: item_id == entity_id (KGAT convention)
    → KHÔNG cần item2entity mapping, KHÔNG re-index
    → assert max(all_item_ids) < n_items
  - valid.txt thay cho val.txt
  - Hỗ trợ kg_type: full / category / brand / none
    (từ KGAT kg_final.txt, không build lại KG)

Format kg_final.txt (KGAT): entity_id  relation_id  entity_id
"""
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from datasets.cf_dataset import CFDataset


class KGDataset(CFDataset):
    def __init__(
        self,
        data_dir:    str,
        split:       str = "train",
        neg_samples: int = 1,
        kg_type:     str = "full",
        seed:        int = 42,
    ) -> None:
        self.kg_type     = kg_type
        self.n_entities: int = 0
        self.n_relations: int = 0
        self.kg_triples: Optional[np.ndarray] = None
        super().__init__(
            data_dir=data_dir, split=split,
            neg_samples=neg_samples, seed=seed,
        )

    def load(self) -> None:
        super().load()
        if self.kg_type != "none":
            self._load_kg()
        # [v10] Kiểm tra bất biến KGAT
        self._assert_kgat_invariant()

    def _assert_kgat_invariant(self) -> None:
        """
        [v10] BẤT BIẾN BẮT BUỘC: item_id == entity_id.
        max(all_item_ids) < n_items theo KGAT convention.
        Nếu fail → pipeline bị lỗi nghiêm trọng.
        """
        from utils.logger import get_logger
        _logger = get_logger("kg_dataset")

        if self.n_items == 0:
            return

        # Lấy tất cả item IDs từ interaction data
        all_item_ids: Set[int] = set()
        for fname in ["train.txt", "valid.txt", "test.txt"]:
            fpath = os.path.join(self.data_dir, fname)
            d = self.read_interaction_file(fpath)
            for items in d.values():
                all_item_ids.update(items)

        if all_item_ids:
            max_item_id = max(all_item_ids)
            assert max_item_id < self.n_items, (
                f"[KGAT INVARIANT VIOLATION] max(item_id)={max_item_id} >= "
                f"n_items={self.n_items}. "
                f"Re-indexing đã làm sai lệch item_id == entity_id!"
            )
            _logger.info(
                f"[KGAT-Invariant] ✓ max(item_id)={max_item_id} < "
                f"n_items={self.n_items} — Bất biến đúng.")

    def _load_kg(self) -> None:
        """
        [v10] Load kg_final.txt từ unified/ (KGAT repo).
        KHÔNG build lại KG — dùng trực tiếp như KGAT repo.
        BẤT BIẾN: item_id == entity_id cho n_items thực thể đầu tiên.
        """
        from utils.logger import get_logger
        _logger = get_logger("kg_dataset")

        kg_file = self._resolve_kg_file()
        if kg_file is None:
            _logger.warning(
                f"Không tìm thấy KG file trong {self.data_dir}. "
                "KG models sẽ chạy không có KG (degraded mode)."
            )
            return

        triples: List[Tuple[int, int, int]] = []
        with open(kg_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    parts = line.strip().split("\t")
                if len(parts) == 3:
                    try:
                        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                        triples.append((h, r, t))
                    except ValueError:
                        continue

        if not triples:
            _logger.warning(f"KG file rỗng: {kg_file}")
            return

        self.kg_triples  = np.array(triples, dtype=np.int64)
        all_entities     = set(self.kg_triples[:, 0]) | set(self.kg_triples[:, 2])
        self.n_entities  = max(all_entities) + 1
        self.n_relations = int(self.kg_triples[:, 1].max()) + 1

        _logger.info(
            f"KG loaded ({self.kg_type} / {os.path.basename(kg_file)}): "
            f"{len(triples):,} triples | "
            f"{self.n_entities:,} entities | "
            f"{self.n_relations:,} relations"
        )

        # [v10] Kiểm tra bất biến KGAT sau khi load KG
        if self.n_items > 0 and self.n_entities > 0:
            if self.n_entities < self.n_items:
                _logger.warning(
                    f"n_entities ({self.n_entities}) < n_items ({self.n_items}). "
                    "Một số items không có trong KG — kiểm tra lại dữ liệu."
                )
            else:
                from utils.logger import get_logger
                _logger.info(
                    f"[KGAT-Invariant] n_items={self.n_items} thực thể đầu "
                    f"của KG (n_entities={self.n_entities}) là item entities."
                )

    def _resolve_kg_file(self) -> Optional[str]:
        """
        [v10] Tìm kg file theo thứ tự ưu tiên:
          1. kg_final.txt (KGAT format — ưu tiên)
          2. kg_final_{type}.txt nếu có (subset)
          3. kg.txt (fallback cũ)
        """
        # Ưu tiên: kg_final.txt từ KGAT (thường là full KG)
        primary = os.path.join(self.data_dir, "kg_final.txt")
        if os.path.exists(primary):
            return primary

        # Fallback cho subset ablation
        kg_type_map = {
            "full":      "kg_final.txt",
            "category":  "kg_category.txt",
            "brand":     "kg_brand.txt",
        }
        if self.kg_type in kg_type_map:
            candidate = os.path.join(self.data_dir, kg_type_map[self.kg_type])
            if os.path.exists(candidate):
                return candidate

        # Legacy fallback (chỉ kg.txt — không dùng kg_full.txt nữa từ v10)
        legacy_path = os.path.join(self.data_dir, "kg.txt")
        if os.path.exists(legacy_path):
            return legacy_path

        return None

    # ── Graph builders ────────────────────────────────────────────────────────

    def build_kg_adj_list(self) -> Dict[int, List[Tuple[int, int]]]:
        """Build adjacency list: head → [(tail, relation), ...]"""
        adj: Dict[int, List[Tuple[int, int]]] = {}
        if self.kg_triples is None:
            return adj
        for h, r, t in self.kg_triples:
            adj.setdefault(int(h), []).append((int(t), int(r)))
        return adj

    def build_kg_norm_adj(self) -> torch.Tensor:
        """
        Xây dựng D^{-1/2}AD^{-1/2} KG adjacency (sparse COO, CPU).
        Dùng cho KGCL và KGLightGCN.
        """
        if self.kg_triples is None:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0),
                (self.n_entities, self.n_entities),
            ).coalesce()

        heads = self.kg_triples[:, 0]
        tails = self.kg_triples[:, 2]
        data  = np.ones(len(heads), dtype=np.float32)
        A     = sp.csr_matrix(
            (data, (heads, tails)),
            shape=(self.n_entities, self.n_entities),
        )
        A = (A + A.T).tocsr()
        deg = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(
                deg > 0, np.power(deg, -0.5), 0.0,
            ).astype(np.float32)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack([A_hat.row, A_hat.col]).astype(np.int64))
        values  = torch.from_numpy(A_hat.data)
        return torch.sparse_coo_tensor(
            indices, values,
            (self.n_entities, self.n_entities),
        ).coalesce()

    # ── KG triple sampling ────────────────────────────────────────────────────

    def sample_kg_triples(
        self, batch_size: int
    ) -> Optional[Tuple]:
        if self.kg_triples is None:
            return None
        idxs     = self.rng.randint(0, len(self.kg_triples), size=batch_size)
        selected = self.kg_triples[idxs]
        heads    = selected[:, 0].tolist()
        rels     = selected[:, 1].tolist()
        t_pos    = selected[:, 2].tolist()
        t_neg    = self.rng.randint(0, self.n_entities, size=batch_size)
        collision = t_neg == selected[:, 2]
        t_neg[collision] = (t_neg[collision] + 1) % self.n_entities
        return heads, rels, t_pos, t_neg.tolist()

    def load_item_category_map(self) -> Dict[int, int]:
        """
        [v10] Load item_category.txt để lấy flat category cho Setting A.
        Format: item_id  category_id
        Trả về {item_id → category_id}.
        """
        fpath = os.path.join(self.data_dir, "item_category.txt")
        mapping: Dict[int, int] = {}
        if not os.path.exists(fpath):
            return mapping
        with open(fpath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        mapping[int(parts[0])] = int(parts[1])
                    except ValueError:
                        continue
        return mapping
