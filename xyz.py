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
