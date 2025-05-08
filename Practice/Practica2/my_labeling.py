__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
import time
import matplotlib.pyplot as plt
from Kmeans import *
from KNN import *
import numpy as np


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
    # Sample data if it's too large (reduces computation time)
    if images.shape[0] > 100000:
        np.random.seed(42)  # For reproducibility
        idx = np.random.choice(images.shape[0], 10000, replace=False)
        print(f"Sampling 10,000 points from {len(images)} for faster computation")
        images = images[idx]
    
    ks = list(range(2, kmax + 1))
    wcds, iters, times = [], [], []
    
    for k in ks:
        print(f'Running K-means with K={k}')
        # Set explicit options to make convergence faster
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
        
        print(f"  - Completed in {elapsed:.2f}s, {itr} iterations, WCD: {wcd:.2f}")

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
    color_tp = {}  # True positives by color
    color_fp = {}  # False positives by color
    color_fn = {}  # False negatives by color
    
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
            
        if verbose:
            print(f"Image {i}: Pred: {pset}, True: {tset}, Score: {score:.2f}")
    
    # Calculate overall accuracy
    avg_accuracy = (sum(scores) / len(scores)) * 100 if scores else 0.0
    
    # Calculate per-color F1 scores
    color_stats = {}
    for color in all_colors:
        precision = color_tp[color] / (color_tp[color] + color_fp[color]) if (color_tp[color] + color_fp[color]) > 0 else 0
        recall = color_tp[color] / (color_tp[color] + color_fn[color]) if (color_tp[color] + color_fn[color]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
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
        for color in sorted(all_colors):
            stats = color_stats[color]
            print(f"  {color:6}: F1={stats['f1']:.2f}, Precision={stats['precision']:.2f}, Recall={stats['recall']:.2f}")
        
        print("\nMost common confusion patterns:")
        confusion_pairs = []
        for pred_color in color_confusion:
            for true_color, count in color_confusion[pred_color].items():
                confusion_pairs.append((pred_color, true_color, count))
        
        # Display top confusions
        for pred, true, count in sorted(confusion_pairs, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  {pred:6} predicted as {true:6}: {count} times")
    
    return avg_accuracy, color_stats


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(
            root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    #
    #
    ############################################
    print('Read the dataset')

    # K-means diagnostics on training set pixels

    print(
        f" Running K-means diagnostics for K = 2…kmax (this can take a while) ")
    all_train_pixels = np.vstack([img.reshape(-1, 3) for img in train_imgs])
    ks, wcds, its, times = kmean_statistics(KMeans, all_train_pixels, kmax=4)

    # 3) Color tagging on test set
    print(" Starting color-tag prediction on test set")
    pred_test_color = []
    for img in test_imgs:
        pix = img.reshape(-1, 3)
        km = KMeans(pix, K=3)
        km.fit()
        # map centroids→color names
        cent_cols = km.centroids
        labels = get_colors(cent_cols)
        pred_test_color.append(set(labels))

    # Unpack the tuple returned by get_color_accuracy
    color_acc, color_stats = get_color_accuracy(pred_test_color, test_color_labels,verbose=False)
    print(f"Test‐set color accuracy: {color_acc:.1f}%")

    # Shape classification on test set (via KNN)

    print("=== Starting shape classification on test set ===")
    knn = KNN(train_imgs, train_class_labels)
    pred_test_shape = knn.predict(test_imgs, k=3)
    shape_acc = get_shape_accuracy(pred_test_shape, test_class_labels)
    print(f"Test‐set shape accuracy: {shape_acc:.1f}%")
