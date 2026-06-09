import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_path = "outputs/phase5_train/results_table_best_by_pair.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Create descriptive labels
    df["label"] = df.apply(lambda r: f"Clust: {r['clustering_representation']} | Class: {r['classifier_representation']} ({r['model_alias'].upper()})", axis=1)
    
    # Sort by Accuracy to make the chart look structured
    df = df.sort_values(by="accuracy", ascending=True)
    
    # Set the style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    # Create a figure with 2 subplots side-by-side or stacked
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    
    # Plot Accuracy
    sns.barplot(
        x="accuracy",
        y="label",
        data=df,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Classification Accuracy", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel("Accuracy Score", fontsize=12)
    axes[0].set_ylabel("Configuration (Clustering / Classifier Mode)", fontsize=12)
    axes[0].set_xlim(0, 1.0)
    
    # Add values on the bars for Accuracy
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f"{width:.4f}",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='semibold'
        )

    # Plot Macro F1
    sns.barplot(
        x="f1_macro",
        y="label",
        data=df,
        palette="magma",
        ax=axes[1]
    )
    axes[1].set_title("Classification Macro F1-Score", fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel("Macro F1-Score", fontsize=12)
    axes[1].set_ylabel("") # Hide y label for the second subplot
    axes[1].set_xlim(0, 1.0)
    
    # Add values on the bars for Macro F1
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f"{width:.4f}",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='semibold'
        )

    plt.tight_layout()
    
    # Save locally in the workspace
    os.makedirs("scratch", exist_ok=True)
    local_img_path = "scratch/phase5_best_runs.png"
    plt.savefig(local_img_path, dpi=300, bbox_inches='tight')
    print(f"Saved local image to: {local_img_path}")
    
    # Save to the artifacts folder in Windows path via WSL mount
    artifact_dir = "/mnt/c/Users/trica/.gemini/antigravity/brain/f58a6514-6ab5-4d9a-9831-11aee45683c6"
    if os.path.exists(artifact_dir):
        artifact_img_path = os.path.join(artifact_dir, "phase5_best_runs.png")
        plt.savefig(artifact_img_path, dpi=300, bbox_inches='tight')
        print(f"Saved artifact image to: {artifact_img_path}")
    else:
        print(f"Warning: Artifact directory {artifact_dir} not mounted or not found from WSL.")

if __name__ == "__main__":
    main()
