from collections import Counter


def calculate_oob_score(
    n_samples,
    bootstrap_indices,
    predictions,
    y_true
):
    training_sets = [
        set(indices) for indices in bootstrap_indices
    ]

    correct = 0
    evaluated = 0

    for i in range(n_samples):
        # Yalnızca i. örneği eğitiminde kullanmayan modeller oy verir.
        votes = [
            predictions[j][i]
            for j in range(len(training_sets))
            if i not in training_sets[j]
        ]

        # OOB oyu olmayan örnek değerlendirmeye alınmaz.
        if not votes:
            continue

        counts = Counter(votes)

        # En çok oy alan etiket; eşitlikte en küçük etiket.
        winner = min(
            counts,
            key=lambda label: (-counts[label], label)
        )

        correct += int(winner == y_true[i])
        evaluated += 1

    if evaluated == 0:
        return 0.0

    return round(float(correct / evaluated), 4)