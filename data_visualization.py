import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    # Load Data
    df = pd.read_csv('train.csv')
    
    # Separate Features and Target
    features = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    target = 'eyeDetection'
    
    X = df[features]
    y = df[target]
    
    # Standardize the data (Important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to reduce to 2 Dimensions
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for the plot
    pca_df = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Target'] = y
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Principal Component 1', 
        y='Principal Component 2', 
        hue='Target', 
        data=pca_df, 
        palette={0: '#3498db', 1: '#e74c3c'}, # Matching project colors
        alpha=0.6,
        s=15
    )
    
    plt.title('2D PCA Projection of EEG Data (Problem Space Visualization)', fontsize=16, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)', fontsize=12)
    
    # Legend mapping
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Eye Open (0)', 'Eye Closed (1)'], title='Eye State')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_file = 'problem_formulation_pca.png'
    plt.savefig(output_file, dpi=300)
    print(output_file)
    
except Exception as e:
    print(f"Error: {e}")
