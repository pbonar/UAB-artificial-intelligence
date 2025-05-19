__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
import time
import matplotlib.pyplot as plt
from Kmeans_improved import *
from KNN_improved import *

import numpy as np
import os

# ===== 4.1 - Qualitative analysis functions =====


def retrieve_by_color(images, color_tags, query_colors, color_percentages=None, true_tags=None):
    """
    Retrieve images that match one or more specified color tags and calculate retrieval statistics.

    Args:
        images (list): List of images.
        color_tags (list of sets/lists): Tags assigned to each image.
        query_colors (str or list): Color or list of colors to search for.
        color_percentages (list of dicts, optional): Optional color frequency data per image.
        true_tags (list of sets, optional): Ground truth color tags for statistical comparison.

    Returns:
        tuple: (matched_images, stats_dict) where matched_images is the filtered list
    """
    if isinstance(query_colors, str):
        query_colors = [query_colors]

    query_set = set(query_colors)
    matched = []
    matched_indices = []

    for i, tags in enumerate(color_tags):
        tags_set = set(tags)
        if query_set.issubset(tags_set):
            matched.append(
                (images[i], color_percentages[i] if color_percentages else 1.0))
            matched_indices.append(i)

    # Sort by color proportion if available
    if color_percentages:
        matched.sort(key=lambda x: -sum(x[1].get(q, 0) for q in query_colors))

    # Calculate statistics if ground truth is provided
    stats = {}
    if true_tags is not None:
        found_count = len(matched)

        true_positives = sum(
            1 for i in matched_indices if query_set.issubset(set(true_tags[i])))

        true_count = sum(
            1 for tags in true_tags if query_set.issubset(set(tags)))

        precision = true_positives / found_count if found_count > 0 else 0
        recall = true_positives / true_count if true_count > 0 else 0

        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        stats = {
            'found_count': found_count,
            'true_count': true_count,
            'true_positives': true_positives,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'summary': f"Found {found_count} images containing {query_colors} (of {true_count} in ground truth)"
        }

        # per-color statistics for multi-color queries
        if len(query_colors) > 1:
            color_stats = {}
            for color in query_colors:
                single_true_count = sum(
                    1 for tags in true_tags if color in tags)
                single_found_count = sum(
                    1 for tags in [color_tags[i] for i in matched_indices] if color in tags)
                single_true_positives = sum(
                    1 for i in matched_indices if color in true_tags[i] and color in color_tags[i])

                color_precision = single_true_positives / single_found_count if single_found_count > 0 else 0
                color_recall = single_true_positives / single_true_count if single_true_count > 0 else 0

                color_stats[color] = {
                    'precision': color_precision * 100,
                    'recall': color_recall * 100
                }
            stats['color_specific'] = color_stats

    return [img for img, _ in matched], stats


def retrieve_by_shape(images, shape_tags, query_shape, shape_confidences=None):
    """
    Retrieve images that match a specified shape label.

    Args:
        images (list): List of images.
        shape_tags (list): Shape label assigned to each image.
        query_shape (str): Desired shape label.
        shape_confidences (list of floats, optional): Optional confidence scores.

    Returns:
        list: Filtered list of images matching the shape query.
    """
    matched = []
    for i, tag in enumerate(shape_tags):
        # print(f"[Shape Retrieval] Image {i} tag: {tag}")
        if tag == query_shape:
            # print(f"[Shape Retrieval] -> Match for {query_shape} in image {i}")
            matched.append(
                (images[i], shape_confidences[i] if shape_confidences else 1.0))

    if shape_confidences:
        matched.sort(key=lambda x: -x[1])
        return [img for img, _ in matched]
    else:
        return [img for img, _ in matched]


def retrieve_by_shape(images, shape_tags, query_shape, shape_confidences=None, true_tags=None):
    """
    Retrieve images that match a specified shape label and calculate retrieval statistics.

    Args:
        images (list): List of images.
        shape_tags (list): Shape label assigned to each image.
        query_shape (str): Desired shape label.
        shape_confidences (list of floats, optional): Optional confidence scores.
        true_tags (list, optional): Ground truth shape labels for statistical comparison.

    Returns:
        tuple: (matched_images, stats_dict) where matched_images is the filtered list
    """
    matched = []
    matched_indices = []

    for i, tag in enumerate(shape_tags):
        if tag == query_shape:
            matched.append(
                (images[i], shape_confidences[i] if shape_confidences else 1.0))
            matched_indices.append(i)

    # Sort by confidence if available
    if shape_confidences:
        matched.sort(key=lambda x: -x[1])

    # Calculate statistics if ground truth is provided
    stats = {}
    if true_tags is not None:
        # Total matching the query in our predictions
        found_count = len(matched)

        true_count = sum(1 for tag in true_tags if tag == query_shape)

        true_positives = sum(
            1 for i in matched_indices if true_tags[i] == query_shape)

        precision = true_positives / found_count if found_count > 0 else 0
        recall = true_positives / true_count if true_count > 0 else 0

        stats = {
            'found_count': found_count,
            'true_count': true_count,
            'true_positives': true_positives,
            'precision': precision * 100,
            'recall': recall * 100,
            'summary': f"Found {found_count} images labeled as '{query_shape}' (of {true_count} in ground truth)"
        }

    return [img for img, _ in matched], stats


def retrieve_combined(images, shape_tags, color_tags, query_shape, query_colors,
                      shape_confidences=None, color_percentages=None):
    """
    Retrieve images that match both a specific shape and a set of color tags.

    Args:
        images (list): List of images.
        shape_tags (list): Predicted shape labels.
        color_tags (list of sets/lists): Predicted color tags.
        query_shape (str): Shape to look for.
        query_colors (str or list): Color(s) to look for.
        shape_confidences (list of floats, optional): Shape confidence values.
        color_percentages (list of dicts, optional): Color proportion data.

    Returns:
        list: Filtered list of images matching both shape and color queries.
    """
    if isinstance(query_colors, str):
        query_colors = [query_colors]

    matched = []
    for i, (shape, colors) in enumerate(zip(shape_tags, color_tags)):
        if shape != query_shape:
            continue
        if all(c in colors for c in query_colors):
            shape_score = shape_confidences[i] if shape_confidences else 1.0
            color_score = sum(color_percentages[i].get(
                c, 0) for c in query_colors) if color_percentages else 1.0
            total_score = shape_score * color_score
            matched.append((images[i], total_score))

    matched.sort(key=lambda x: -x[1])
    return [img for img, _ in matched]

# ===== 4.2 - Quantitative analysis functions =====


def kmean_statistics(kmeans_class, images, kmax):
    """
    For K values from 2 to kmax, fits K-means on the provided images,
    collects Within-Class Distance (WCD), number of iterations to converge,
    and elapsed time, then visualizes each metric versus K.

    Parameters:
    - kmeans_class: class implementing Kmeans with methods fit(), withinClassDistance(), and an attribute iterations_
    - images: array-like of image data (e.g., pixel matrices stacked)
    - kmax: int, maximum K to analyze

    Returns:
    - ks: list of K values
    - wcds: list of WCD per K
    - iters: list of iteration counts per K
    - times: list of fit durations per K (seconds)
    """
    ks = list(range(2, kmax + 1))
    wcds, iters, times = [], [], []

    for k in ks:
        print(f'Running K-means with K={k}')
        options = {'km_init': 'random', 'max_iter': 100, 'tolerance': 1e-3}
        km = kmeans_class(images, K=k, options=options)

        # Time the execution
        start = time.time()
        km.fit()
        elapsed = time.time() - start

        # Collect metrics
        wcd = km.withinClassDistance()

        # Ensure iterations are being tracked
        itr = km.num_iter

        wcds.append(wcd)
        iters.append(itr)
        times.append(elapsed)

        print(
            f"  - Completed in {elapsed:.2f}s, {itr} iterations, WCD: {wcd:.2f}")

    # Create a combined figure with all three plots
    plt.figure(figsize=(15, 5))

    # Plot WCD vs K
    plt.subplot(1, 3, 1)
    plt.plot(ks, wcds, 'o-', color='blue')
    plt.title("Within-Class Distance vs K")
    plt.xlabel("K")
    plt.ylabel("WCD")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Iterations vs K
    plt.subplot(1, 3, 2)
    plt.plot(ks, iters, 'o-', color='green')
    plt.title("Iterations to Converge vs K")
    plt.xlabel("K")
    plt.ylabel("Iterations")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Time vs K
    plt.subplot(1, 3, 3)
    plt.plot(ks, times, 'o-', color='red')
    plt.title("Time to Converge vs K")
    plt.xlabel("K")
    plt.ylabel("Time (s)")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return ks, wcds, iters, times


def get_shape_accuracy(pred_tags, true_tags):
    """
    Computes percentage of exact matches between predicted shape tags and ground truth.

    Parameters:
    - pred_tags: list of predicted tags (one per image)
    - true_tags: list of ground-truth tags (one per image)

    Returns:
    - accuracy: float, percentage of correctly predicted tags
    """
    total = len(true_tags)
    correct = sum(p == t for p, t in zip(pred_tags, true_tags))
    return (correct / total) * 100 if total else 0.0


def get_color_accuracy(pred_tag_sets, true_tag_sets, verbose=True):
    """
    Computes average Jaccard similarity between predicted and true color-tag sets,
    with detailed diagnostics on color prediction accuracy.

    Parameters:
    - pred_tag_sets: list of iterable predicted tag sets (per image)
    - true_tag_sets: list of iterable true tag sets (per image)
    - verbose: whether to print detailed diagnostics (default: True)

    Returns:
    - avg_jaccard: float, average Jaccard index (0-100%)
    - color_stats: dict with per-color accuracy statistics
    """
    scores = []

    # For tracking per-color accuracy:
    color_confusion = {}
    color_tp = {}
    color_fp = {}
    color_fn = {}

    # Initialize counters for all colors in utils.colors
    all_colors = set()
    for true in true_tag_sets:
        all_colors.update(true)
    for pred in pred_tag_sets:
        all_colors.update(pred)

    for color in all_colors:
        color_tp[color] = 0
        color_fp[color] = 0
        color_fn[color] = 0
        color_confusion[color] = {}

    # Main evaluation loop
    for i, (pred, true) in enumerate(zip(pred_tag_sets, true_tag_sets)):
        pset, tset = set(pred), set(true)

        # Calculate Jaccard similarity
        union = pset | tset
        inter = pset & tset
        score = len(inter) / len(union) if union else 0
        scores.append(score)

        # Update per-color statistics
        for color in inter:
            color_tp[color] += 1
        for color in pset - tset:  # Colors predicted but not in truth
            color_fp[color] += 1
            # Track what was predicted instead of truth
            for true_color in tset:
                if true_color not in color_confusion[color]:
                    color_confusion[color][true_color] = 0
                color_confusion[color][true_color] += 1

        for color in tset - pset:  # Colors in truth but not predicted
            color_fn[color] += 1

    avg_accuracy = (sum(scores) / len(scores)) * 100 if scores else 0.0

    # Calculate per-color F1 scores
    color_stats = {}
    for color in all_colors:
        precision = color_tp[color] / (color_tp[color] + color_fp[color]) if (
            color_tp[color] + color_fp[color]) > 0 else 0
        recall = color_tp[color] / (color_tp[color] + color_fn[color]
                                    ) if (color_tp[color] + color_fn[color]) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0
        color_stats[color] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion': color_confusion[color]
        }

    if verbose:
        print("\n--- Color Accuracy Statistics ---")
        print(f"Overall accuracy: {avg_accuracy:.1f}%")
        print("\nPer-color metrics:")
        print(f"{'Color':6} |  | {'Precision':>10} | {'Recall':>10} | {'F1':>6}")
        for color in sorted(all_colors):
            stats = color_stats[color]
            print(f"  {color:6}: {stats['precision']:.2f}, {stats['recall']:.2f}, {stats['f1']:.2f}")

    return avg_accuracy, color_stats


def plot_color_confusion_matrix(pred_tag_sets, true_tag_sets):
    """
    Creates a direct confusion matrix for color classification focusing on errors.

    Parameters:
    - pred_tag_sets: list of sets containing predicted colors for each image
    - true_tag_sets: list of sets containing true colors for each image

    Returns:
    - confusion_matrix: numpy array with the confusion counts
    """
    all_colors = set()
    for true in true_tag_sets:
        all_colors.update(true)
    for pred in pred_tag_sets:
        all_colors.update(pred)

    all_colors = sorted(all_colors)
    n_colors = len(all_colors)

    color_to_idx = {color: i for i, color in enumerate(all_colors)}

    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_colors, n_colors), dtype=int)

    # Fill the confusion matrix
    for true_set, pred_set in zip(true_tag_sets, pred_tag_sets):
        true_set = set(true_set)
        pred_set = set(pred_set)

        # Add to diagonal for correct predictions
        for color in true_set.intersection(pred_set):
            idx = color_to_idx[color]
            confusion_matrix[idx, idx] += 1

        # For each true color that wasn't predicted (false negatives)
        # Count what colors were predicted instead
        for missed_color in true_set - pred_set:
            true_idx = color_to_idx[missed_color]
            # Distribute the miss across the predicted colors
            if pred_set:  # Only if there were predictions
                for wrong_color in pred_set:
                    pred_idx = color_to_idx[wrong_color]
                    confusion_matrix[true_idx, pred_idx] += 1 / len(pred_set)

        # For each predicted color that wasn't in ground truth (false positives)
        for extra_color in pred_set - true_set:
            pred_idx = color_to_idx[extra_color]
            # Distribute the false positive across the true colors
            if true_set:  # Only if there were true colors
                for actual_color in true_set:
                    true_idx = color_to_idx[actual_color]
                    confusion_matrix[true_idx, pred_idx] += 1 / len(true_set)

    # Round to integers after distributing fractional counts
    confusion_matrix = np.round(confusion_matrix).astype(int)

    # Visualization
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.Blues

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Direct Color Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(n_colors)
    plt.xticks(tick_marks, all_colors, rotation=45, ha='right')
    plt.yticks(tick_marks, all_colors)

    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(n_colors):
        for j in range(n_colors):
            plt.text(j, i, confusion_matrix[i, j],
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Color')
    plt.xlabel('Predicted Color')
    plt.show()

    return confusion_matrix


def filter_predicted_colors(percentages, max_colors=2, min_percent=0.15):
    """
    Returns up to max_colors color labels with at least min_percent share.
    If not enough colors meet the threshold, fills up to max_colors with the most dominant.
    """
    sorted_colors = sorted(percentages.items(), key=lambda x: -x[1])
    filtered = [color for color, perc in sorted_colors if perc >= min_percent]
    # Always take at least the most dominant color
    if not filtered and sorted_colors:
        filtered = [sorted_colors[0][0]]
    # Limit to max_colors
    filtered = filtered[:max_colors]
    return set(filtered)

# ====== 4.3 - Experiments with color functions =====
def run_color_thresholds_experiment():
    """
    Test different thresholds for minimal color percentage and print accuracy for each.
    """
    thresholds = [x / 100 for x in range(0, 55, 5)]  # 0.00, 0.05, ..., 0.50
    print("Testing color thresholds for minimal color percentage:")
    # Wczytaj dane jak w main
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json')
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_test_imgs = crop_images(test_imgs, upper, lower)
    for thresh in thresholds:
        pred_test_color = []
        for img in cropped_test_imgs:
            pix = img.reshape(-1, 3)
            unique_pix = np.unique(pix, axis=0)
            if len(unique_pix) < 2:
                pred_test_color.append({'White'})
                continue
            K = min(6, max(2, len(unique_pix)))
            km = KMeans(pix, K=K)
            km.fit()
            cent_cols = km.centroids
            labels = get_colors(cent_cols)
            pixel_count = len(pix)
            cluster_counts = np.bincount(km.labels, minlength=km.K)
            percentages = {labels[i]: cluster_counts[i] / pixel_count for i in range(km.K)}
            dominant_colors = set([color for color, perc in sorted(percentages.items(), key=lambda x: -x[1]) if perc >= thresh])
            if not dominant_colors and percentages:
                dominant_colors = {max(percentages, key=percentages.get)}
            dominant_colors = set(list(dominant_colors)[:2])
            pred_test_color.append(dominant_colors)
        acc, _ = get_color_accuracy(pred_test_color, test_color_labels, verbose=False)
        print(f"Threshold: {int(thresh*100):2d}%  ->  Accuracy: {acc:.1f}%")

def test_color_thresholds():
    """
    Pytest-compatible test function for threshold experiment.
    Prints accuracy for thresholds from 0% to 50% (step 5%).
    """
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json')
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_test_imgs = crop_images(test_imgs, upper, lower)
    thresholds = [x / 100 for x in range(0, 55, 5)]  # 0.00, 0.05, ..., 0.50
    print("Testing color thresholds for minimal color percentage:")
    for thresh in thresholds:
        pred_test_color = []
        for img in cropped_test_imgs:
            pix = img.reshape(-1, 3)
            unique_pix = np.unique(pix, axis=0)
            if len(unique_pix) < 2:
                pred_test_color.append({'White'})
                continue
            K = min(6, max(2, len(unique_pix)))
            km = KMeans(pix, K=K)
            km.fit()
            cent_cols = km.centroids
            labels = get_colors(cent_cols)
            pixel_count = len(pix)
            cluster_counts = np.bincount(km.labels, minlength=km.K)
            percentages = {labels[i]: cluster_counts[i] / pixel_count for i in range(km.K)}
            dominant_colors = set([color for color, perc in sorted(percentages.items(), key=lambda x: -x[1]) if perc >= thresh])
            if not dominant_colors and percentages:
                dominant_colors = {max(percentages, key=percentages.get)}
            dominant_colors = set(list(dominant_colors)[:2])
            pred_test_color.append(dominant_colors)
        acc, _ = get_color_accuracy(pred_test_color, test_color_labels, verbose=False)
        print(f"Threshold: {int(thresh*100):2d}%  ->  Accuracy: {acc:.1f}%")

def test_k_vs_accuracy():
    """
    Pytest-compatible test function for K vs. accuracy with threshold 5%.
    Prints accuracy for K=2..8.
    """
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json')
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_test_imgs = crop_images(test_imgs, upper, lower)
    min_percent = 0.05
    print("Testing K vs. accuracy (threshold 5%):")
    for K in range(2, 9):
        pred_test_color = []
        for img in cropped_test_imgs:
            pix = img.reshape(-1, 3)
            unique_pix = np.unique(pix, axis=0)
            if len(unique_pix) < 2:
                pred_test_color.append({'White'})
                continue
            k_val = min(K, max(2, len(unique_pix)))
            km = KMeans(pix, K=k_val)
            km.fit()
            cent_cols = km.centroids
            labels = get_colors(cent_cols)
            pixel_count = len(pix)
            cluster_counts = np.bincount(km.labels, minlength=km.K)
            percentages = {labels[i]: cluster_counts[i] / pixel_count for i in range(km.K)}
            dominant_colors = set([color for color, perc in sorted(percentages.items(), key=lambda x: -x[1]) if perc >= min_percent])
            if not dominant_colors and percentages:
                dominant_colors = {max(percentages, key=percentages.get)}
            dominant_colors = set(list(dominant_colors)[:2])
            pred_test_color.append(dominant_colors)
        acc, _ = get_color_accuracy(pred_test_color, test_color_labels, verbose=False)
        print(f"K = {K}: Accuracy = {acc:.1f}%")

def test_k_and_threshold_grid():
    """
    Pytest-compatible test function for K and threshold grid search.
    Prints accuracy for all combinations K=2..8 and threshold=0%,5%,...,50%.
    """
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json')
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_test_imgs = crop_images(test_imgs, upper, lower)
    thresholds = [x / 100 for x in range(0, 55, 5)]  # 0.00, 0.05, ..., 0.50
    print("Grid search: K vs. threshold (accuracy):")
    print("      " + "  ".join([f"thr={int(t*100):2d}%" for t in thresholds]))
    for K in range(2, 9):
        accs = []
        for thresh in thresholds:
            pred_test_color = []
            for img in cropped_test_imgs:
                pix = img.reshape(-1, 3)
                unique_pix = np.unique(pix, axis=0)
                if len(unique_pix) < 2:
                    pred_test_color.append({'White'})
                    continue
                k_val = min(K, max(2, len(unique_pix)))
                km = KMeans(pix, K=k_val)
                km.fit()
                cent_cols = km.centroids
                labels = get_colors(cent_cols)
                pixel_count = len(pix)
                cluster_counts = np.bincount(km.labels, minlength=km.K)
                percentages = {labels[i]: cluster_counts[i] / pixel_count for i in range(km.K)}
                dominant_colors = set([color for color, perc in sorted(percentages.items(), key=lambda x: -x[1]) if perc >= thresh])
                if not dominant_colors and percentages:
                    dominant_colors = {max(percentages, key=percentages.get)}
                dominant_colors = set(list(dominant_colors)[:2])
                pred_test_color.append(dominant_colors)
            acc, _ = get_color_accuracy(pred_test_color, test_color_labels, verbose=False)
            accs.append(f"{acc:5.1f}")
        print(f"K={K}: " + "  ".join(accs))


# ===== Main function =====
if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth and cropped images
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    print('Read the dataset')

    # # --- Metrics ---
    # print("Running K-means diagnostics for K = 2…kmax (this can take a while)")
    # all_train_pixels = np.vstack([img.reshape(-1, 3) for img in train_imgs])
    # ks, wcds, its, times = kmean_statistics(KMeans, all_train_pixels, kmax=6)

    print("Starting color-tag prediction on test set")
    pred_test_color = []
    color_percentages = []
    cropped_test_imgs = crop_images(test_imgs, upper, lower)
    for img in cropped_test_imgs:
        pix = img.reshape(-1, 3)
        unique_pix = np.unique(pix, axis=0)
        if len(unique_pix) < 2:
            pred_test_color.append({'White'})
            color_percentages.append({'White': 1.0})
            continue
        # Ustal K=6 i threshold=0.10 (10%)
        K = min(6, max(2, len(unique_pix)))
        km = KMeans(pix, K=K)
        km.fit()
        cent_cols = km.centroids
        labels = get_colors(cent_cols)
        pixel_count = len(pix)
        cluster_counts = np.bincount(km.labels, minlength=km.K)
        percentages = {labels[i]: cluster_counts[i] / pixel_count for i in range(km.K)}
        color_percentages.append(percentages)
        dominant_colors = set([color for color, perc in sorted(percentages.items(), key=lambda x: -x[1]) if perc >= 0.10])
        if not dominant_colors and percentages:
            dominant_colors = {max(percentages, key=percentages.get)}
        dominant_colors = set(list(dominant_colors)[:2])
        pred_test_color.append(dominant_colors)

    color_acc, color_stats = get_color_accuracy(
        pred_test_color, test_color_labels, verbose=True)
    print(f"Test set color accuracy: {color_acc:.1f}%")


    conf_matrix = plot_color_confusion_matrix(
        pred_test_color, test_color_labels)

    print("Starting shape classification on test set")
    knn = KNN(train_imgs, train_class_labels)
    pred_test_shape = knn.predict(test_imgs, k=3)
    shape_acc = get_shape_accuracy(pred_test_shape, test_class_labels)
    print(f"Test set shape accuracy: {shape_acc:.1f}%")

    print("Searching for test images containing 'Red' and 'Blue' colors...")

    results_color, color_stats = retrieve_by_color(test_imgs, pred_test_color, ['Red', 'Blue'],
                                                   color_percentages=color_percentages, true_tags=test_color_labels)
    print(color_stats['summary'])
    print(f"Precision: {color_stats['precision']:.1f}%, Recall: {color_stats['recall']:.1f}%")
    print(f"F1-Score: {color_stats['f1_score']:.1f}%")

    # For multi-color queries, you can also access per-color statistics
    if 'color_specific' in color_stats:
        print("\nPer-color statistics:")
        for color, stats in color_stats['color_specific'].items():
            print(f"  {color}: Precision: {stats['precision']:.1f}%, Recall: {stats['recall']:.1f}%")

        for i, (img, tags) in enumerate(zip(test_imgs, pred_test_color)):
            if all(c in tags for c in ['Red', 'Blue']):
                print(f"  - Matching file: test_{i}.jpg with tags: {tags}")

    print("Searching for test images predicted as 'Jeans'...")

    results_shape, stats = retrieve_by_shape(test_imgs, pred_test_shape, 'Jeans',
                                             true_tags=test_class_labels)
    print(stats['summary'])
    print(f"Precision: {stats['precision']:.1f}%, Recall: {stats['recall']:.1f}%")

    for i, tag in enumerate(pred_test_shape):
        if tag == 'Jeans':
            print(f"  - Matching file: test_{i}.jpg with shape: {tag}")

    print("Searching for 'Blue Jeans' images...")
    results_combined = retrieve_combined(
        test_imgs, pred_test_shape, pred_test_color, 'Tshirt', 'White')
    print(f"Found {len(results_combined)} images matching combined query.")
    for i, (tag, colors) in enumerate(zip(pred_test_shape, pred_test_color)):
        if tag == 'Jeans' and 'Blue' in colors:
            print(
                f"  - Matching file: test_{i}.jpg with shape: {tag} and colors: {colors}")

    if results_combined:
        print("Displaying all 'Blue Jeans' results in a grid...")
        max_width_in = 16  # max width of figure in inches
        img_width = 3  # individual image width in inches
        cols = max(1, min(5, max_width_in // img_width))
        rows = math.ceil(len(results_combined) / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(img_width * cols, 3 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for ax in axes[len(results_combined):]:
            ax.axis('off')

        for idx, (img, ax) in enumerate(zip(results_combined, axes)):
            ax.imshow(img.astype(np.uint8))
            ax.set_title(f"Image {idx}")
            ax.axis('off')

        plt.tight_layout()
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')  # Maximize window on Windows
        except:
            try:
                # Resize to max on Linux/macOS
                mng.resize(*mng.window.maxsize())
            except:
                pass
        plt.show()

        print("Comparing distance metrics for KNN shape classification")

        for metric in ['euclidean', 'manhattan']:
            print(f"\nUsing distance metric: {metric}")

            knn = KNN(train_imgs, train_class_labels, distance_metric=metric)
            pred_test_shape = knn.predict(test_imgs, k=3)

            shape_acc = get_shape_accuracy(pred_test_shape, test_class_labels)
            print(f"Test-set shape accuracy with {metric}: {shape_acc:.1f}%")

        print("Distance metric vs. k comparison")

        for metric in ['euclidean', 'manhattan']:
            print(f"\nMetric: {metric}")
            for k in [1, 3, 5, 7]:
                knn = KNN(train_imgs, train_class_labels, distance_metric=metric)
                pred_test_shape = knn.predict(test_imgs, k=k)
                acc = get_shape_accuracy(pred_test_shape, test_class_labels)
                print(f"k = {k}: Accuracy = {acc:.1f}%")

        heuristics = ['wcd', 'inter_class', 'fisher']
        print("Comparing K selection heuristics for color classification")

        for heuristic in heuristics:
            print(f"\nUsing heuristic: {heuristic}")
            pred_test_color = []

            for img in test_imgs:
                pix = img.reshape(-1, 3)

                # Starte mit einem Dummy-K (z.B. 2), wird gleich überschrieben
                km = KMeans(pix, K=2, options={"km_init": "random"})

                # Bestes K bestimmen (maximal z. B. 6)
                best_K = km.find_bestK_by_heuristic(max_K=6, method=heuristic, verbose=False)

                # KMeans neu initialisieren mit bestem K
                km = KMeans(pix, K=best_K, options={"km_init": "random"})
                km.fit()

                cent_cols = km.centroids
                labels = get_colors(cent_cols)
                pred_test_color.append(set(labels))

            color_acc, color_stats = get_color_accuracy(pred_test_color, test_color_labels, verbose=False)
            print(f"Color accuracy using '{heuristic}': {color_acc:.1f}%")

        from collections import Counter

        heuristics = ['wcd', 'inter_class', 'fisher']
        print(
            "Comparing global best K for each heuristic")  # this takes a while! That's why it is in comments right now

        # for heuristic in heuristics:
        #    print(f"Heuristic: '{heuristic}'")

        #    # ---------------------------------------
        #    # 1. GLOBAL BEST K on training data
        #    # ---------------------------------------
        #    all_pixels = []

        #    for img in train_imgs:
        #        pix = img.reshape(-1, 3)
        #        all_pixels.append(pix)

        #    all_pixels = np.concatenate(all_pixels, axis=0)

        #    km = KMeans(all_pixels, K=2, options={"km_init": "random"})
        #    best_K = km.find_bestK_by_heuristic(max_K=6, method=heuristic, verbose=False)

        #    print(f"Global best K determined on training set: {best_K}")

        # ---------------------------------------
        # 2. TEST with global K
        # ---------------------------------------
        #    pred_test_color = []

        #    for img in test_imgs:
        #        pix = img.reshape(-1, 3)
        #        km = KMeans(pix, K=best_K, options={"km_init": "random"})
        #        km.fit()
        #        centroids = km.centroids
        #        labels = get_colors(centroids)
        #        pred_test_color.append(set(labels))

        # ---------------------------------------
        # 3. Color Accuracy
        # ---------------------------------------
        #    color_acc, _ = get_color_accuracy(pred_test_color, test_color_labels, verbose=False)
        #    print(f"Color accuracy using global K: {color_acc:.1f}%")