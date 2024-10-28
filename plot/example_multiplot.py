import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x**2
y4 = np.exp(-x/5)

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sine Wave', 'Cosine Wave', 'Quadratic', 'Exponential'),
    specs=[[{'type': 'xy'}, {'type': 'xy'}],
           [{'type': 'xy'}, {'type': 'xy'}]]
)

# Add traces with unique ids
fig.add_trace(
    go.Scatter(x=x, y=y1, name="sin(x)", line=dict(color='blue'), ids=['plot1']),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y2, name="cos(x)", line=dict(color='red'), ids=['plot2']),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=x, y=y3, name="x²", line=dict(color='green'), ids=['plot3']),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y4, name="exp(-x/5)", line=dict(color='purple'), ids=['plot4']),
    row=2, col=2
)

# Create visibility buttons
def create_visibility_buttons():
    buttons = [
        # Show all plots
        dict(
            label='Show All',
            method='update',
            args=[{'visible': [True, True, True, True]},
                  {'showlegend': True}]
        ),
        # Hide all plots
        dict(
            label='Hide All',
            method='update',
            args=[{'visible': [False, False, False, False]},
                  {'showlegend': True}]
        ),
        # Show only top row
        dict(
            label='Show Top Row',
            method='update',
            args=[{'visible': [True, True, False, False]},
                  {'showlegend': True}]
        ),
        # Show only bottom row
        dict(
            label='Show Bottom Row',
            method='update',
            args=[{'visible': [False, False, True, True]},
                  {'showlegend': True}]
        ),
        # Show only left column
        dict(
            label='Show Left Column',
            method='update',
            args=[{'visible': [True, False, True, False]},
                  {'showlegend': True}]
        ),
        # Show only right column
        dict(
            label='Show Right Column',
            method='update',
            args=[{'visible': [False, True, False, True]},
                  {'showlegend': True}]
        )
    ]
    
    # Create individual toggle buttons for each plot
    for i, plot_name in enumerate(['Sine', 'Cosine', 'Quadratic', 'Exponential']):
        visibility = [False] * 4  # Start with all hidden
        visibility[i] = True      # Show only this plot
        buttons.append(
            dict(
                label=f'Show {plot_name}',
                method='update',
                args=[{'visible': visibility},
                      {'showlegend': True}]
            )
        )
    
    return buttons

# Update layout with buttons
fig.update_layout(
    updatemenus=[
        dict(
            buttons=create_visibility_buttons(),
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )
    ],
    # Add a title to the figure
    title_text="Interactive Subplot Visibility Control",
    height=800,
    width=1000,
    showlegend=True,
)

# Update axes labels
fig.update_xaxes(title_text="X")
fig.update_yaxes(title_text="Y")

# Show figure
fig.show()