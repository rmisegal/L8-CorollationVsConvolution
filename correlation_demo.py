import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class CorrelationVisualizer:
    """
    Interactive 2D correlation visualization showing how correlation coefficient
    affects scatter plot patterns.
    """
    
    def __init__(self):
        self.correlation = 0.0
        self.n_points = 200
        self.setup_figure()
        
    def setup_figure(self):
        """Setup the correlation visualization figure."""
        self.fig, (self.ax_scatter, self.ax_info) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Setup scatter plot
        self.ax_scatter.set_title('2D Correlation Visualization', fontsize=16, fontweight='bold')
        self.ax_scatter.set_xlim(-4, 4)
        self.ax_scatter.set_ylim(-4, 4)
        self.ax_scatter.grid(True, alpha=0.3)
        self.ax_scatter.set_xlabel('X Variable', fontsize=14)
        self.ax_scatter.set_ylabel('Y Variable', fontsize=14)
        self.ax_scatter.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_scatter.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Setup info panel
        self.ax_info.set_title('Correlation Information', fontsize=16, fontweight='bold')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        # Add slider
        slider_ax = plt.axes([0.15, 0.02, 0.6, 0.03])
        self.slider = Slider(slider_ax, 'Correlation Coefficient', -1.0, 1.0, 
                           valinit=0.0, valfmt='%.3f', facecolor='lightblue')
        self.slider.on_changed(self.update_correlation)
        
        # Add regenerate button
        from matplotlib.widgets import Button
        button_ax = plt.axes([0.8, 0.02, 0.15, 0.04])
        self.btn_regenerate = Button(button_ax, 'New Data')
        self.btn_regenerate.on_clicked(self.regenerate_data)
        
        # Initial plot
        self.update_plot()
        
    def generate_correlated_data(self, correlation):
        """Generate 2D data with specified correlation."""
        # Generate independent random variables
        x1 = np.random.randn(self.n_points)
        x2 = np.random.randn(self.n_points)
        
        # Create correlated variables using Cholesky decomposition
        # Correlation matrix
        corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
        
        # Cholesky decomposition (handle edge cases)
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # For perfect correlation cases
            L = np.array([[1.0, 0.0], [correlation, np.sqrt(1 - correlation**2)]])
        
        # Generate correlated data
        uncorrelated = np.vstack([x1, x2])
        correlated = L @ uncorrelated
        
        return correlated[0], correlated[1]
        
    def update_correlation(self, val):
        """Update correlation value and refresh plot."""
        self.correlation = self.slider.val
        self.update_plot()
        
    def regenerate_data(self, event=None):
        """Generate new random data with current correlation."""
        self.update_plot()
        
    def update_plot(self):
        """Update the scatter plot and information panel."""
        # Clear previous plot
        self.ax_scatter.clear()
        self.ax_info.clear()
        
        # Reapply scatter plot settings
        self.ax_scatter.set_title('2D Correlation Visualization', fontsize=16, fontweight='bold')
        self.ax_scatter.set_xlim(-4, 4)
        self.ax_scatter.set_ylim(-4, 4)
        self.ax_scatter.grid(True, alpha=0.3)
        self.ax_scatter.set_xlabel('X Variable', fontsize=14)
        self.ax_scatter.set_ylabel('Y Variable', fontsize=14)
        self.ax_scatter.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_scatter.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Generate and plot correlated data
        x_data, y_data = self.generate_correlated_data(self.correlation)
        
        # Color points based on correlation strength
        if abs(self.correlation) < 0.3:
            color = 'blue'
            alpha = 0.6
        elif abs(self.correlation) < 0.7:
            color = 'orange'
            alpha = 0.7
        else:
            color = 'red'
            alpha = 0.8
            
        self.ax_scatter.scatter(x_data, y_data, c=color, alpha=alpha, s=40, 
                              edgecolors='black', linewidth=0.5)
        
        # Add correlation value text
        self.ax_scatter.text(0.02, 0.98, f'r = {self.correlation:.3f}', 
                           transform=self.ax_scatter.transAxes, fontsize=18, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
        
        # Add trend line for strong correlations
        if abs(self.correlation) > 0.5:
            # Calculate and plot trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(-4, 4, 100)
            y_trend = p(x_trend)
            # Clip to plot bounds
            valid_indices = (y_trend >= -4) & (y_trend <= 4)
            self.ax_scatter.plot(x_trend[valid_indices], y_trend[valid_indices], 
                               "r--", alpha=0.8, linewidth=2, label=f'Trend line')
        
        # Update info panel
        self.ax_info.set_title('Correlation Information', fontsize=16, fontweight='bold')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        # Correlation interpretation
        if abs(self.correlation) < 0.1:
            interpretation = "No linear relationship"
            pattern = "Random circular scatter"
        elif abs(self.correlation) < 0.3:
            interpretation = "Weak linear relationship"
            pattern = "Slight elliptical pattern"
        elif abs(self.correlation) < 0.7:
            interpretation = "Moderate linear relationship"
            pattern = "Clear elliptical pattern"
        elif abs(self.correlation) < 0.9:
            interpretation = "Strong linear relationship"
            pattern = "Narrow elliptical pattern"
        else:
            interpretation = "Very strong linear relationship"
            pattern = "Nearly perfect line"
            
        direction = "Positive (upward)" if self.correlation > 0 else "Negative (downward)" if self.correlation < 0 else "None"
        
        # Calculate actual correlation
        actual_r = np.corrcoef(x_data, y_data)[0,1]
        
        info_text = f"""Correlation Coefficient: {self.correlation:.3f}

Strength: {interpretation}
Direction: {direction}
Pattern: {pattern}

Key Correlation Values:
• r = +1.0: Perfect positive correlation (45° upward line)
• r = +0.7: Strong positive correlation
• r = +0.3: Weak positive correlation
• r =  0.0: No linear correlation (circle)
• r = -0.3: Weak negative correlation
• r = -0.7: Strong negative correlation
• r = -1.0: Perfect negative correlation (45° downward line)

Current Dataset:
• Sample size: {self.n_points} points
• Mean X: {np.mean(x_data):.2f}
• Mean Y: {np.mean(y_data):.2f}
• Std X: {np.std(x_data):.2f}
• Std Y: {np.std(y_data):.2f}
• Actual r: {actual_r:.3f}

Instructions:
• Drag the slider to change correlation
• Click 'New Data' to regenerate points
• Observe how patterns change from circle to line"""
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        # Refresh display
        self.fig.canvas.draw()
        
    def show(self):
        """Display the correlation visualization."""
        plt.show()

def main():
    """Main function to run the correlation visualizer."""
    print("=" * 60)
    print("2D Correlation Visualization Demo")
    print("=" * 60)
    print("\nThis interactive demo shows how correlation affects scatter plots:")
    print("• Drag the slider to change correlation from -1 to +1")
    print("• Watch how the scatter pattern changes:")
    print("  - r = 0: Random circular pattern")
    print("  - r = +1: Perfect 45° upward line")
    print("  - r = -1: Perfect 45° downward line")
    print("• Click 'New Data' to generate fresh random points")
    print("• Red dashed line shows trend for strong correlations")
    print("\nEducational Value:")
    print("• Understand correlation vs causation")
    print("• See how correlation strength affects scatter patterns")
    print("• Learn to interpret correlation coefficients")
    print("=" * 60)
    
    visualizer = CorrelationVisualizer()
    visualizer.show()

if __name__ == "__main__":
    main()
