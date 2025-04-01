import json
import matplotlib.pyplot as plt
import os

def compute_rates(question_logs_file):
    with open(question_logs_file, "r") as f:
        data = json.load(f)

    correctness_rates = []

    for i, batch in enumerate(data):
        if i >= 300:
            break

        false_negatives = 0
        false_positives = 0
        true_positives = 0
        true_negatives = 0
        batch_size = 0
        # Go through each question in the batch
        for q_num in batch:
            if q_num.startswith('q'):  # Only process question entries
                question = batch[q_num]
                if question["gold_score"] == 0.0:
                if question["gold_score"] == 0.0:
                    continue
                batch_size += 1
                if (question['extracted_judge_score'] == 0) and (question['gold_score'] > 0.5):
                    false_negatives += 1
                elif (question['extracted_judge_score'] == 1) and (question['gold_score'] < 0.5):
                    false_positives += 1
                elif (question['extracted_judge_score'] == 0) and (question['gold_score'] < 0.5):
                    true_negatives += 1
                elif (question['extracted_judge_score'] == 1) and (question['gold_score'] > 0.5):
                    true_positives += 1
        
        # Calculate false negative rate for this batch
        if batch_size > 0:
            fn_rate = false_negatives / batch_size
            fp_rate = false_positives / batch_size
            tp_rate = true_positives / batch_size
            tn_rate = true_negatives / batch_size
            fn_rate = false_negatives / batch_size
            fp_rate = false_positives / batch_size
            tp_rate = true_positives / batch_size
            tn_rate = true_negatives / batch_size
            
            correctness_rates.append({
                'batch_size': batch_size,
                'false_negatives': false_negatives,
                'false_negative_rate': fn_rate,
                'false_positives': false_positives,
                'false_positive_rate': fp_rate,
                'true_positives': true_positives,
                'true_positive_rate': tp_rate,
                'true_negatives': true_negatives,
                'true_negative_rate': tn_rate
            })

    # Write false negative rates to JSON file
    with open("correctness_rates.json", "w") as f:
        json.dump(correctness_rates, f, indent=4)
    
    return correctness_rates
    
def plot_correctness_metrics(correctness_rates, file_path):
    steps = list(range(1, len(correctness_rates) + 1))
    batch_sizes = [item['batch_size'] for item in correctness_rates]
    false_negatives = [item['false_negative_rate'] for item in correctness_rates]
    false_positives = [item['false_positive_rate'] for item in correctness_rates]
    true_positives = [item['true_positive_rate'] for item in correctness_rates]
    true_negatives = [item['true_negative_rate'] for item in correctness_rates]

    plt.figure(figsize=(10, 5))
    #plt.plot(steps, batch_sizes, label='Batch Size')
    #plt.plot(steps, batch_sizes, label='Batch Size')
    plt.plot(steps, false_negatives, label='False Negatives')
    plt.plot(steps, false_positives, label='False Positives')
    plt.plot(steps, true_positives, label='True Positives')
    plt.plot(steps, true_negatives, label='True Negatives')
    plt.xlabel('Steps')
    plt.ylabel('Rate')
    plt.title('Correctness Metrics for LLM Judge')
    plt.legend()

    plt.savefig(file_path)
    plt.close()

    # Plot FP / (FP + TN)
    fp_rates = [item['false_positive_rate'] / (item['false_positive_rate'] + item['true_negative_rate'] + 1e-10) for item in correctness_rates]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, fp_rates, label='FP / (FP + TN)')
    plt.xlabel('Steps')
    plt.ylabel('Rate')
    plt.title('False Positives / (False Positives + True Negatives)')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(file_path), 'fp_rates.png'))

    # Plot FN / (FN + TP)
    fn_rates = [item['false_negative_rate'] / (item['false_negative_rate'] + item['true_positive_rate'] + 1e-10) for item in correctness_rates]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, fn_rates, label='FN / (FN + TP)')
    plt.xlabel('Steps')
    plt.ylabel('Rate')
    plt.title('False Negatives / (False Negatives + True Positives)')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(file_path), 'fn_rates.png'))

if __name__ == "__main__":
    question_logs_file = "/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/svg_judge_experiments/qwen2.5_7b_svg_gpt4o_mini/question_logs.json"
    plot_file_path = "/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/svg_judge_experiments/qwen2.5_7b_svg_gpt4o_mini/correctness_metrics.png"
    
    correctness_rates = compute_rates(question_logs_file)
    plot_correctness_metrics(correctness_rates, plot_file_path)