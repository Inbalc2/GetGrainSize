import statistics

def compute_statistics(data, label):
    """Compute and print statistics for a given dataset."""
    mean_value = statistics.mean(data)
    median_value = statistics.median(data)
    try:
        mode_value = statistics.mode(data)
    except statistics.StatisticsError:
        mode_value = "No unique mode"
    standard_deviation = statistics.stdev(data)
    variance = statistics.variance(data)
    min_value = min(data)
    max_value = max(data)
    sum_value = sum(data)

    print(f"{label}:")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Mode: {mode_value}")
    print(f"Standard Deviation: {standard_deviation}")
    print(f"Variance: {variance}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}")
    print(f"Sum: {sum_value}")
    print("-----------------------------------")

def main(df_path):
    df = pd.read_csv(df_path)
    models = df['Model'].unique()
    
    for model in models:
        data = df[df['Model'] == model]['Mean Grain Size'].tolist()
        compute_statistics(data, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, help='Path to the CSV file containing the results')
    args = parser.parse_args()
    main(args.df_path)
