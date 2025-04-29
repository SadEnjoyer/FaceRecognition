def calculate_tar_at_fars(scores, labels, target_fars=[0.001, 0.01, 0.05, 0.1]): #Target_FARS можно адаптировать, сделал для 0.1,1,5,10
    scores = np.array(scores)
    labels = np.array(labels)
    thresholds = np.linspace(0, 1, 10000)

    results = {}

    for target_far in target_fars:
        best_tar = 0
        best_threshold = 0
        best_far = 1

        for thresh in thresholds:
            preds = (scores > thresh).astype(int) * 2 - 1

            tp = np.sum((preds == 1) & (labels == 1))
            fn = np.sum((preds == -1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == -1))
            tn = np.sum((preds == -1) & (labels == -1))

            tar = tp / (tp + fn) if (tp + fn) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0

            if abs(far - target_far) < abs(best_far - target_far):
                best_tar = tar
                best_far = far
                best_threshold = thresh

        results[f"{int(target_far*100)}%"] = {
            "TAR": best_tar,
            "FAR": best_far,
            "threshold": best_threshold
        }
        print(f"TAR: {best_tar:.4f}, FAR: {best_far:.4f} @ threshold = {best_threshold:.4f} (target FAR = {target_far*100:.1f}%)")

    return results