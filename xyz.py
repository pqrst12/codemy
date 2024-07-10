import plotly.graph_objs as go

def plot_3d_target_vs_features(feature1, feature2, target):
    fig = go.Figure(data=[go.Scatter3d(
        x=feature1,
        y=feature2,
        z=target,
        mode='markers',
        marker=dict(
            size=8,
            color=target,                # Color by target variable
            colorscale='Viridis',       # Choose a colorscale
            opacity=0.8
        )
    )])

    # Update layout
    fig.update_layout(scene=dict(
                        xaxis_title='Feature 1',
                        yaxis_title='Feature 2',
                        zaxis_title='Target Variable'),
                      title='3D Plot of Target Variable vs. Features')
    
    fig.show()




X, y = make_classification(n_samples=30000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform the data
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Class 0', alpha=0.6)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='Class 1', alpha=0.6)
plt.legend()
plt.title('t-SNE visualization of binary classification data')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()

