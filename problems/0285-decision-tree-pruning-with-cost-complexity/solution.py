def compute_pruning_alphas(tree: dict) -> list:
    alphas = []

    def traverse(node):
        if node is None:
            return 0.0, 0

        # Yaprak: kendi hata sayısını ve 1 yaprak döndür
        if node["left"] is None and node["right"] is None:
            return node["errors"], 1

        # Sol ve sağ alt ağacın bilgilerini hesapla
        left_errors, left_leaves = traverse(node["left"])
        right_errors, right_leaves = traverse(node["right"])

        subtree_errors = left_errors + right_errors
        leaf_count = left_leaves + right_leaves

        # İç düğümün etkin alfa değerini hesapla
        if leaf_count > 1:
            alpha = (
                (node["errors"] - subtree_errors)
                / (leaf_count - 1)
            )
            alphas.append(float(alpha))

        # Üst düğümün hesabaında kullanılacak bilgiler
        return subtree_errors, leaf_count

    traverse(tree)
    return sorted(alphas)