from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def find_top_n(embeddings,n,index,data):
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    top_ids_and_scores = index.search_knn(embeddings, n)
    data_ans=[]
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        data_now=[]
        for id in ids:
            data_now.append((data[0][int(id)],data[1][int(id)],data[2][int(id)]))
        data_ans.append(data_now)
    return data_ans


def print_line(class_name, metrics, is_header=False):
    if is_header:
        line = f"| {'Class':<10} | " + " | ".join([f"{metric:<10}" for metric in metrics])
    else:
        line = f"| {class_name:<10} | " + " | ".join([f"{metrics[metric]:<10.3f}" for metric in metrics])
    print(line)
    if is_header:
        print('-' * len(line))

def calculate_per_class_metrics(classes, ground_truth, predictions):
    # Convert ground truth and predictions to numeric format
    gt_numeric = np.array([int(gt) for gt in ground_truth])
    pred_numeric = np.array([int(pred) for pred in predictions])

    results = {}
    for i, class_name in enumerate(classes):
        # For each class, calculate the 'vs rest' binary labels
        gt_binary = (gt_numeric == i).astype(int)
        pred_binary = (pred_numeric == i).astype(int)

        # Calculate metrics, handling cases where a class is not present in predictions or ground truth
        precision = precision_score(gt_binary, pred_binary, zero_division=0)
        recall = recall_score(gt_binary, pred_binary, zero_division=0)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)
        acc = np.mean(gt_binary == pred_binary)
        # Calculate recall for all other classes as 'rest'
        rest_recall = recall_score(1 - gt_binary, 1 - pred_binary, zero_division=0)

        results[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': acc,
            'Avg Recall (with rest)': (recall + rest_recall) / 2
        }

    print_line("Metric", results[classes[0]], is_header=True)
    for class_name, metrics in results.items():
        print_line(class_name, metrics)
    overall_metrics = {metric_name: np.mean([metrics[metric_name] for metrics in results.values()]) for metric_name in results[classes[0]].keys()}
    print_line("Overall", overall_metrics)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    avg_f1 = f1_score(y_true, y_pred, average='macro')
    avg_recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, avg_f1,avg_recall

def compute_three_recalls(labels, preds):
    all_n, all_p, tn, tp = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == '0':
            all_p += 1
        if label == '1':
            all_n += 1
        if pred is not None and label == pred == '0':
            tp += 1 
        if pred is not None and label == pred == '1':
            tn += 1
        if pred is None:
            continue
    machine_rec , human_rec= tp * 100 / all_p if all_p != 0 else 0, tn * 100 / all_n if all_n != 0 else 0
    avg_rec = (human_rec + machine_rec) / 2
    return (human_rec, machine_rec, avg_rec)


def compute_metrics(labels, preds,ids=None):
    if ids is not None:
        # unique ids
        dict_labels,dict_preds={},{}
        for i in range(len(ids)):
            dict_labels[ids[i]]=labels[i]
            dict_preds[ids[i]]=preds[i] 
        labels=list(dict_labels.values())
        preds=list(dict_preds.values())
    
    human_rec, machine_rec, avg_rec = compute_three_recalls(labels, preds)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label='1')
    recall = recall_score(labels, preds, pos_label='1')
    f1 = f1_score(labels, preds, pos_label='1')
    # return human_rec, machine_rec, avg_rec
    return (human_rec, machine_rec, avg_rec, acc, precision, recall, f1)