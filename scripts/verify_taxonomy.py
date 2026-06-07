"""
scripts/verify_taxonomy.py — v10
Xác minh cấu trúc taxonomy trong kg_final.txt (KGAT Amazon-Book).
Chạy script này để confirm Setting B không suy biến về Setting A.

Usage:
  python scripts/verify_taxonomy.py \
      --kg_path /data/phuongtran/project_v10/unified/amazon-book/kg_final.txt \
      --n_items 24915
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("verify_taxonomy")


def analyze_kg_hierarchy(kg_path: str, n_items: int) -> dict:
    """
    Phân tích cấu trúc hierarchy trong KG:
    1. Phân loại entity types (item vs non-item)
    2. Tìm relation nào kết nối categories với nhau (hierarchy indicator)
    3. Đo depth của category tree
    """
    if not os.path.exists(kg_path):
        logger.error(f"Không tìm thấy: {kg_path}")
        return {}

    # Load triples
    logger.info(f"Đang đọc {kg_path} ...")
    item_to_ent:   dict = defaultdict(set)   # item → {entity}
    ent_to_ent:    dict = defaultdict(set)   # non-item-entity → {entity}
    rel_counts:    dict = defaultdict(int)
    head_entity:   set  = set()
    tail_entity:   set  = set()

    with open(kg_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            rel_counts[r] += 1
            head_entity.add(h)
            tail_entity.add(t)

            if h < n_items:
                item_to_ent[h].add(t)   # item → non-item
            else:
                ent_to_ent[h].add(t)    # non-item → non-item (hierarchy!)

            if i % 500_000 == 0 and i > 0:
                logger.info(f"  {i:,} triples đã đọc...")

    logger.info(f"  Tổng: {sum(rel_counts.values()):,} triples")

    # Phân tích
    all_entities = head_entity | tail_entity
    n_entities   = max(all_entities) + 1

    item_entities   = {e for e in all_entities if e < n_items}
    non_item_ents   = {e for e in all_entities if e >= n_items}

    # KEY: Non-item → non-item relations = HIERARCHY SIGNAL
    hierarchy_triples = sum(len(v) for v in ent_to_ent.values())
    hierarchy_rels    = set()
    with open(kg_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                if h >= n_items and t >= n_items:
                    hierarchy_rels.add(r)
            except ValueError:
                pass

    # Chiều sâu tối đa của category tree (BFS/DFS từ leaf)
    max_depth = 0
    try:
        if ent_to_ent:
            # Tìm root entities (không phải tail của entity-entity relation)
            all_heads_ent = set(ent_to_ent.keys())
            all_tails_ent = set()
            for heads in ent_to_ent.values():
                all_tails_ent.update(heads)
            
            leaf_ents = all_heads_ent - all_tails_ent  # heads that aren't tails
            
            # BFS để đo depth
            from collections import deque
            visited = {}
            queue = deque()
            
            # Start từ một sample
            sample_leaves = list(leaf_ents)[:100]
            for leaf in sample_leaves:
                queue.append((leaf, 0))
                visited[leaf] = 0
            
            while queue:
                node, depth = queue.popleft()
                max_depth = max(max_depth, depth)
                for neighbor in ent_to_ent.get(node, set()):
                    if neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
    except Exception as e:
        logger.warning(f"BFS error: {e}")

    # Item-category connectivity
    items_with_nonitem_link = len(item_to_ent)
    avg_nonitem_per_item    = (
        sum(len(v) for v in item_to_ent.values()) / max(items_with_nonitem_link, 1)
    )

    results = {
        "n_entities":                n_entities,
        "n_relations":               len(rel_counts),
        "n_item_entities":           len(item_entities),
        "n_non_item_entities":       len(non_item_ents),
        "n_hierarchy_triples":       hierarchy_triples,
        "n_hierarchy_relations":     len(hierarchy_rels),
        "hierarchy_relation_ids":    sorted(hierarchy_rels)[:20],
        "max_category_depth_approx": max_depth,
        "items_with_nonitem_link":   items_with_nonitem_link,
        "avg_nonitem_per_item":      round(avg_nonitem_per_item, 2),
        "top_relations_by_count": sorted(
            rel_counts.items(), key=lambda x: x[1], reverse=True)[:15],
    }
    return results


def print_taxonomy_report(results: dict, n_items: int) -> None:
    print("\n" + "=" * 65)
    print("TAXONOMY HIERARCHY ANALYSIS — KGAT Amazon-Book")
    print("=" * 65)

    n_hier_triples = results.get("n_hierarchy_triples", 0)
    n_hier_rels    = results.get("n_hierarchy_relations", 0)
    max_depth      = results.get("max_category_depth_approx", 0)

    print(f"\n  n_entities:             {results.get('n_entities', 0):,}")
    print(f"  n_item_entities:        {results.get('n_item_entities', 0):,}")
    print(f"  n_non_item_entities:    {results.get('n_non_item_entities', 0):,}")
    print(f"\n  HIERARCHY INDICATORS:")
    print(f"    non-item→non-item triples: {n_hier_triples:,}")
    print(f"    hierarchy relation IDs:    {n_hier_rels} unique relations")
    print(f"    relation IDs:              {results.get('hierarchy_relation_ids', [])}")
    print(f"    estimated max depth:       {max_depth}")
    print(f"\n  ITEM-KG CONNECTIVITY:")
    print(f"    items with non-item links: {results.get('items_with_nonitem_link', 0):,}")
    print(f"    avg non-item entities/item: {results.get('avg_nonitem_per_item', 0)}")

    print(f"\n  TOP RELATIONS BY FREQUENCY:")
    for rel_id, count in results.get("top_relations_by_count", [])[:10]:
        pct = count / max(sum(c for _, c in results["top_relations_by_count"]), 1) * 100
        print(f"    rel_{rel_id:>3d}: {count:>10,} triples ({pct:.1f}%)")

    print("\n" + "=" * 65)
    print("VERDICT — Setting B (Taxonomy-guided CL):")
    print("=" * 65)

    if n_hier_triples > 0 and n_hier_rels >= 2:
        depth_label = "≥3 cấp" if max_depth >= 2 else f"~{max_depth+1} cấp"
        print(f"\n  ✅ KG CÓ cấu trúc hierarchy ({depth_label})")
        print(f"     {n_hier_triples:,} entity-entity triples với {n_hier_rels} relation types")
        print(f"\n  → Setting B KHÔNG suy biến về Setting A")
        print(f"  → Positive pairs Setting B = items chia sẻ cùng cấp L2/L3 của taxonomy")
        print(f"\n  IMPLEMENTATION:")
        print(f"    hierarchy_rels = {results.get('hierarchy_relation_ids', [])[:5]}")
        print(f"    → Dùng các relation IDs này để extract taxonomy tree")
        print(f"    → Với items: dùng relation kết nối item → category entity")
    elif n_hier_triples == 0:
        print(f"\n  ⚠ CẢNH BÁO: KHÔNG tìm thấy entity-entity triples")
        print(f"     KG chỉ có item-attribute relations (flat)")
        print(f"     → Setting B SẼ suy biến về Setting A")
        print(f"     → Ghi cảnh báo này vào dataset_stats.md và paper")
    else:
        print(f"\n  ⚠ CẦN XEM XÉT THÊM: Ít hierarchy ({n_hier_triples:,} triples)")

    print("=" * 65)


def main():
    p = argparse.ArgumentParser(
        description="Verify taxonomy hierarchy trong KGAT kg_final.txt")
    p.add_argument(
        "--kg_path",
        default="/data/phuongtran/project_v10/unified/amazon-book/kg_final.txt",
    )
    p.add_argument("--n_items", type=int, default=24915)
    p.add_argument("--output",  default="results/taxonomy_analysis.md")
    args = p.parse_args()

    results = analyze_kg_hierarchy(args.kg_path, args.n_items)
    if not results:
        return

    print_taxonomy_report(results, args.n_items)

    # Ghi file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        import json
        f.write("# Taxonomy Analysis — KGAT Amazon-Book\n\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2, default=str))
        f.write("\n```\n")
    logger.info(f"✓ taxonomy_analysis.md → {args.output}")


if __name__ == "__main__":
    main()
