import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches

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
        self.fig, (self.ax_scatter, self.ax_info) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Setup scatter plot
        self.ax_scatter.set_title('2D Correlation Visualization', fontsize=14, fontweight='bold')
        self.ax_scatter.set_xlim(-4, 4)
        self.ax_scatter.set_ylim(-4, 4)
        self.ax_scatter.grid(True, alpha=0.3)
        self.ax_scatter.set_xlabel('X Variable', fontsize=12)
        self.ax_scatter.set_ylabel('Y Variable', fontsize=12)
        self.ax_scatter.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_scatter.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Setup info panel
        self.ax_info.set_title('Correlation Information', fontsize=14, fontweight='bold')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        # Add slider
        slider_ax = plt.axes([0.2, 0.02, 0.5, 0.03])
        self.slider = Slider(slider_ax, 'Correlation', -1.0, 1.0, valinit=0.0, valfmt='%.2f')
        self.slider.on_changed(self.update_correlation)
        
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
        
        # Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate correlated data
        uncorrelated = np.vstack([x1, x2])
        correlated = L @ uncorrelated
        
        return correlated[0], correlated[1]
        
    def update_correlation(self, val):
        """Update correlation value and refresh plot."""
        self.correlation = self.slider.val
        self.update_plot()
        
    def update_plot(self):
        """Update the scatter plot and information panel."""
        # Clear previous plot
        self.ax_scatter.clear()
        self.ax_info.clear()
        
        # Reapply scatter plot settings
        self.ax_scatter.set_title('2D Correlation Visualization', fontsize=14, fontweight='bold')
        self.ax_scatter.set_xlim(-4, 4)
        self.ax_scatter.set_ylim(-4, 4)
        self.ax_scatter.grid(True, alpha=0.3)
        self.ax_scatter.set_xlabel('X Variable', fontsize=12)
        self.ax_scatter.set_ylabel('Y Variable', fontsize=12)
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
            
        self.ax_scatter.scatter(x_data, y_data, c=color, alpha=alpha, s=30, edgecolors='black', linewidth=0.5)
        
        # Add correlation value text
        self.ax_scatter.text(0.02, 0.98, f'r = {self.correlation:.3f}', 
                           transform=self.ax_scatter.transAxes, fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Update info panel
        self.ax_info.set_title('Correlation Information', fontsize=14, fontweight='bold')
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
            
        direction = "Positive (↗)" if self.correlation > 0 else "Negative (↙)" if self.correlation < 0 else "None"
        
        info_text = f"""Correlation Coefficient: {self.correlation:.3f}

Strength: {interpretation}
Direction: {direction}
Pattern: {pattern}

Key Points:
• r = +1: Perfect positive correlation (45° line)
• r = 0: No linear correlation (circle)
• r = -1: Perfect negative correlation (-45° line)

Current Data:
• {self.n_points} random points
• Mean X: {np.mean(x_data):.2f}
• Mean Y: {np.mean(y_data):.2f}
• Actual r: {np.corrcoef(x_data, y_data)[0,1]:.3f}"""
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Refresh display
        self.fig.canvas.draw()
        
    def show(self):
        """Display the correlation visualization."""
        plt.show()

class EnhancedConvolutionVisualizer:
    """
    Enhanced interactive visualization of 1D convolution process.
    Shows how a kernel of length 5 slides over a vector X of length 10.
    Demonstrates proper convolution with zero-padding and mathematical accuracy.
    """
    
    def __init__(self):
        # Initialize vectors
        self.kernel_size = 5
        self.x_size = 10
        self.output_size = self.x_size + self.kernel_size - 1  # Full convolution size
        
        # Generate random values with better control
        np.random.seed(42)  # For reproducible results initially
        self.X = np.random.uniform(-2, 2, self.x_size)  # More controlled range
        self.H_original = np.random.uniform(-1, 1, self.kernel_size)  # Kernel values
        
        # Initialize output vector Y
        self.Y = np.zeros(self.output_size)
        
        # Current step in convolution
        self.current_step = 0
        self.max_steps = self.output_size
        
        # Store computation details for display
        self.current_computation = ""
        
        # Create the figure and subplots
        self.setup_figure()
        
    def setup_figure(self):
        """Setup the matplotlib figure with all subplots and controls."""
        self.fig = plt.figure(figsize=(20, 14))
        
        # Create grid layout with better proportions - now 5 rows
        gs = self.fig.add_gridspec(5, 4, height_ratios=[1.0, 0.8, 1.2, 1.0, 0.3], 
                                  width_ratios=[1, 1, 1, 1])
        
        # Vector displays (top row)
        self.ax_x = self.fig.add_subplot(gs[0, 0])
        self.ax_h = self.fig.add_subplot(gs[0, 1])
        self.ax_y = self.fig.add_subplot(gs[0, 2])
        
        # New sliding window visualization (second row, spans 3 columns)
        self.ax_sliding = self.fig.add_subplot(gs[1, :3])
        
        # Y Vector progress (top right, spans 2 rows)
        self.ax_y_progress = self.fig.add_subplot(gs[0:2, 3])
        
        # 2D vector visualization (third row, spans all columns)
        self.ax_2d = self.fig.add_subplot(gs[2, :])
        
        # Computation details (fourth row, spans all columns)
        self.ax_computation = self.fig.add_subplot(gs[3, :])
        
        # Control buttons (bottom row)
        self.ax_buttons = self.fig.add_subplot(gs[4, :])
        
        self.setup_vector_plots()
        self.setup_sliding_window()
        self.setup_y_progress()
        self.setup_2d_plot()
        self.setup_computation_display()
        self.setup_buttons()
        
        # Initial display
        self.update_display()
        
    def setup_vector_plots(self):
        """Setup the three vector display plots."""
        # X vector plot
        self.ax_x.set_title('Vector X (Input Signal)', fontsize=14, fontweight='bold', pad=20)
        self.ax_x.set_xlim(-0.5, self.x_size - 0.5)
        self.ax_x.set_ylim(-3, 3)
        self.ax_x.grid(True, alpha=0.3)
        self.ax_x.set_xlabel('Index', fontsize=12)
        self.ax_x.set_ylabel('Value', fontsize=12)
        
        # H vector plot (extended with zeros)
        self.ax_h.set_title('Kernel H (Sliding Window)', fontsize=14, fontweight='bold', pad=20)
        self.ax_h.set_xlim(-0.5, self.output_size - 0.5)
        self.ax_h.set_ylim(-2, 2)
        self.ax_h.grid(True, alpha=0.3)
        self.ax_h.set_xlabel('Index', fontsize=12)
        self.ax_h.set_ylabel('Value', fontsize=12)
        
        # Y vector plot
        self.ax_y.set_title('Output Y (Convolution Result)', fontsize=14, fontweight='bold', pad=20)
        self.ax_y.set_xlim(-0.5, self.output_size - 0.5)
        self.ax_y.set_ylim(-8, 8)
        self.ax_y.grid(True, alpha=0.3)
        self.ax_y.set_xlabel('Index', fontsize=12)
        self.ax_y.set_ylabel('Value', fontsize=12)
        
    def setup_sliding_window(self):
        """Setup the sliding window visualization showing H moving over X."""
        self.ax_sliding.set_title('Kernel H Sliding Over Padded Input X', 
                                fontsize=12, fontweight='bold', pad=15)
        # Calculate total padded length
        total_length = self.x_size + 2 * (self.kernel_size - 1)
        self.ax_sliding.set_xlim(-0.5, total_length - 0.5)
        self.ax_sliding.set_ylim(-0.5, 2.5)
        self.ax_sliding.grid(True, alpha=0.3)
        self.ax_sliding.set_xlabel('Position', fontsize=10)
        self.ax_sliding.set_ylabel('Layer', fontsize=10)
        
        # Add labels for layers
        self.ax_sliding.text(-1, 0, 'X (padded)', fontsize=10, ha='right', va='center', fontweight='bold')
        self.ax_sliding.text(-1, 1, 'H (kernel)', fontsize=10, ha='right', va='center', fontweight='bold')
        
    def setup_y_progress(self):
        """Setup the Y vector progress display."""
        self.ax_y_progress.set_title('Y Vector\nProgress', fontsize=12, fontweight='bold', pad=15)
        self.ax_y_progress.set_xlim(0, 1)
        self.ax_y_progress.set_ylim(0, 1)
        self.ax_y_progress.axis('off')
        
    def setup_2d_plot(self):
        """Setup the 2D vector visualization plot."""
        self.ax_2d.set_title('2D Vector: H[0:2] × X_slice[0:2] + Scalar Result', 
                            fontsize=14, fontweight='bold', pad=20)
        self.ax_2d.set_xlim(-6, 6)
        self.ax_2d.set_ylim(-6, 6)
        self.ax_2d.grid(True, alpha=0.3)
        self.ax_2d.set_xlabel('X Component (H[0] × X_slice[0])', fontsize=12)
        self.ax_2d.set_ylabel('Y Component (H[1] × X_slice[1])', fontsize=12)
        self.ax_2d.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_2d.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
    def setup_computation_display(self):
        """Setup the computation details display."""
        self.ax_computation.set_title('Current Computation Details', 
                                    fontsize=14, fontweight='bold', pad=20)
        self.ax_computation.set_xlim(0, 1)
        self.ax_computation.set_ylim(0, 1)
        self.ax_computation.axis('off')
        
    def setup_buttons(self):
        """Setup control buttons."""
        self.ax_buttons.set_xlim(0, 10)
        self.ax_buttons.set_ylim(0, 1)
        self.ax_buttons.axis('off')
        
        # Next step button
        ax_next = plt.axes([0.15, 0.02, 0.12, 0.04])
        self.btn_next = Button(ax_next, 'Next Step')
        self.btn_next.on_clicked(self.next_step)
        
        # Reset button
        ax_reset = plt.axes([0.3, 0.02, 0.12, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)
        
        # Auto play button
        ax_auto = plt.axes([0.45, 0.02, 0.12, 0.04])
        self.btn_auto = Button(ax_auto, 'Auto Play')
        self.btn_auto.on_clicked(self.auto_play)
        
        # New random button
        ax_random = plt.axes([0.6, 0.02, 0.12, 0.04])
        self.btn_random = Button(ax_random, 'New Random')
        self.btn_random.on_clicked(self.new_random)
        
        # Correlation visualizer button
        ax_correlation = plt.axes([0.75, 0.02, 0.15, 0.04])
        self.btn_correlation = Button(ax_correlation, 'Correlation Demo')
        self.btn_correlation.on_clicked(self.show_correlation)
        
    def get_padded_x(self):
        """Get X vector padded with zeros for full convolution."""
        # Pad X with zeros: (kernel_size-1) zeros on each side for full convolution
        padding = self.kernel_size - 1
        return np.pad(self.X, (padding, padding), 'constant', constant_values=0)
        
    def get_extended_kernel(self):
        """Get the kernel H extended with zeros and positioned according to current step."""
        H_extended = np.zeros(self.output_size)
        
        # Position the original kernel based on current step
        start_pos = self.current_step
        end_pos = min(start_pos + self.kernel_size, self.output_size)
        kernel_start = max(0, -start_pos)
        kernel_end = kernel_start + (end_pos - start_pos)
        
        if end_pos > start_pos:
            H_extended[start_pos:end_pos] = self.H_original[kernel_start:kernel_end]
            
        return H_extended
        
    def compute_convolution_step(self):
        """Compute the convolution at current step with detailed breakdown."""
        X_padded = self.get_padded_x()
        
        # Get the slice of X that overlaps with kernel at current position
        start_idx = self.current_step
        end_idx = start_idx + self.kernel_size
        
        if end_idx <= len(X_padded):
            x_slice = X_padded[start_idx:end_idx]
            # For convolution, we flip the kernel
            h_flipped = np.flip(self.H_original)
            
            # Compute dot product
            dot_product = np.dot(x_slice, h_flipped)
            
            # Create detailed computation string
            x_str = " + ".join([f"({x_slice[i]:.2f}×{h_flipped[i]:.2f})" 
                               for i in range(len(x_slice))])
            self.current_computation = f"Y[{self.current_step}] = {x_str} = {dot_product:.3f}"
            
            return dot_product, x_slice, h_flipped
        return 0, np.array([]), np.array([])
        
    def update_display(self):
        """Update all visual elements."""
        # Clear previous plots
        self.ax_x.clear()
        self.ax_h.clear()
        self.ax_y.clear()
        self.ax_sliding.clear()
        self.ax_y_progress.clear()
        self.ax_2d.clear()
        self.ax_computation.clear()
        
        # Reapply settings
        self.setup_vector_plots()
        self.setup_sliding_window()
        self.setup_y_progress()
        self.setup_2d_plot()
        self.setup_computation_display()
        
        # Plot X vector with padding visualization
        X_padded = self.get_padded_x()
        x_full_indices = np.arange(len(X_padded))
        x_colors = ['lightblue' if i < self.kernel_size-1 or i >= len(X_padded)-(self.kernel_size-1) 
                   else 'blue' for i in range(len(X_padded))]
        
        # Show only the original X vector in the X plot
        x_indices = np.arange(self.x_size)
        self.ax_x.bar(x_indices, self.X, alpha=0.8, color='blue', edgecolor='black', linewidth=1)
        for i, val in enumerate(self.X):
            self.ax_x.text(i, val + 0.1 if val >= 0 else val - 0.2, f'{val:.2f}', 
                          ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')
        
        # Plot extended H vector (kernel sliding window)
        H_extended = self.get_extended_kernel()
        h_indices = np.arange(self.output_size)
        colors = ['red' if H_extended[i] != 0 else 'lightgray' for i in range(self.output_size)]
        self.ax_h.bar(h_indices, H_extended, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        for i, val in enumerate(H_extended):
            if val != 0:
                self.ax_h.text(i, val + 0.05 if val >= 0 else val - 0.15, f'{val:.2f}', 
                              ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')
        
        # Highlight current computation area
        if self.current_step < self.max_steps:
            rect = patches.Rectangle((self.current_step - 0.4, -2), 0.8, 4, 
                                   linewidth=3, edgecolor='orange', facecolor='yellow', alpha=0.3)
            self.ax_h.add_patch(rect)
        
        # Plot Y vector (results)
        y_indices = np.arange(self.output_size)
        y_colors = ['green' if i < self.current_step else 'orange' if i == self.current_step else 'lightgray' 
                   for i in range(self.output_size)]
        self.ax_y.bar(y_indices, self.Y, alpha=0.8, color=y_colors, edgecolor='black', linewidth=1)
        for i, val in enumerate(self.Y):
            if i <= self.current_step and val != 0:
                self.ax_y.text(i, val + 0.2 if val >= 0 else val - 0.4, f'{val:.2f}', 
                              ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')
        
        # Plot sliding window visualization
        self.plot_sliding_window()
        
        # Plot Y vector progress
        self.plot_y_progress()
        
        # Plot 2D vector representation
        if len(self.H_original) >= 2:
            # Get current X slice for this convolution step
            X_padded = self.get_padded_x()
            start_idx = self.current_step
            end_idx = start_idx + self.kernel_size
            
            if end_idx <= len(X_padded) and self.current_step < self.max_steps:
                x_slice = X_padded[start_idx:end_idx]
                h_flipped = np.flip(self.H_original)
                
                # Get first two components of H and corresponding X values
                h_x, h_y = self.H_original[0], self.H_original[1]  # First two of original H
                x_x = x_slice[0] if len(x_slice) > 0 else 0  # First X value in current slice
                x_y = x_slice[1] if len(x_slice) > 1 else 0  # Second X value in current slice
                
                # Create 2D vector: H[0:2] × X[current_slice[0:2]]
                vec_x = h_x * x_x
                vec_y = h_y * x_y
                
                # Current convolution result
                current_result = self.Y[self.current_step] if self.current_step < len(self.Y) else 0
                
                # Draw H vector (constant - first two components)
                self.ax_2d.arrow(0, 0, h_x, h_y, head_width=0.15, head_length=0.15, 
                               fc='blue', ec='blue', linewidth=2, alpha=0.6, 
                               label=f'H[0:2] = ({h_x:.2f}, {h_y:.2f})')
                
                # Draw X vector (current slice first two components)
                self.ax_2d.arrow(0, 0, x_x, x_y, head_width=0.15, head_length=0.15, 
                               fc='green', ec='green', linewidth=2, alpha=0.6,
                               label=f'X_slice[0:2] = ({x_x:.2f}, {x_y:.2f})')
                
                # Draw result vector (H[0:2] × X[0:2])
                if vec_x != 0 or vec_y != 0:
                    self.ax_2d.arrow(0, 0, vec_x, vec_y, head_width=0.2, head_length=0.2, 
                                   fc='red', ec='red', linewidth=3,
                                   label=f'H×X = ({vec_x:.2f}, {vec_y:.2f})')
                
                # Add information boxes
                info_text = f'Step: {self.current_step + 1}/{self.max_steps}\n'
                info_text += f'H[0:2]: ({h_x:.2f}, {h_y:.2f})\n'
                info_text += f'X_slice[0:2]: ({x_x:.2f}, {x_y:.2f})\n'
                info_text += f'H×X Vector: ({vec_x:.2f}, {vec_y:.2f})\n'
                info_text += f'Full Dot Product: {current_result:.3f}'
                
                self.ax_2d.text(0.02, 0.98, info_text, transform=self.ax_2d.transAxes, 
                              fontsize=11, verticalalignment='top',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
                
                # Add scalar result at top right to avoid covering vectors
                self.ax_2d.text(0.98, 0.95, f'Scalar Result: {current_result:.3f}', 
                              transform=self.ax_2d.transAxes, fontsize=14, fontweight='bold',
                              ha='right', va='top',
                              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.9))
            else:
                # Show final state or initial state
                self.ax_2d.text(0.5, 0.5, 'Convolution\nComplete!' if self.current_step >= self.max_steps else 'Ready to Start', 
                              transform=self.ax_2d.transAxes, fontsize=16, fontweight='bold',
                              ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
        
    def plot_sliding_window(self):
        """Plot the sliding window showing H moving over padded X."""
        X_padded = self.get_padded_x()
        total_length = len(X_padded)
        
        # Plot padded X vector at y=0
        for i, val in enumerate(X_padded):
            color = 'lightblue' if i < self.kernel_size-1 or i >= total_length-(self.kernel_size-1) else 'blue'
            self.ax_sliding.bar(i, 0.8, bottom=0, alpha=0.7, color=color, edgecolor='black', linewidth=1)
            if val != 0:  # Only show non-zero values
                self.ax_sliding.text(i, 0.4, f'{val:.1f}', ha='center', va='center', fontsize=8, fontweight='bold')
            else:
                self.ax_sliding.text(i, 0.4, '0', ha='center', va='center', fontsize=8, color='gray')
        
        # Plot kernel H at y=1, positioned according to current step
        if self.current_step < self.max_steps:
            kernel_start = self.current_step
            for i, val in enumerate(self.H_original):
                pos = kernel_start + i
                if pos < total_length:
                    self.ax_sliding.bar(pos, 0.8, bottom=1, alpha=0.8, color='red', edgecolor='black', linewidth=1)
                    self.ax_sliding.text(pos, 1.4, f'{val:.1f}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # Add arrow showing current computation position
            arrow_pos = kernel_start + self.kernel_size // 2
            self.ax_sliding.annotate(f'Step {self.current_step + 1}', 
                                   xy=(arrow_pos, 2), xytext=(arrow_pos, 2.3),
                                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                                   arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    def plot_y_progress(self):
        """Plot the Y vector progress display."""
        y_text = "Y Vector Progress:\n\n"
        
        # Show completed values
        for i in range(min(self.current_step, len(self.Y))):
            y_text += f"Y[{i:2d}] = {self.Y[i]:6.3f} [DONE]\n"
        
        # Show current calculation
        if self.current_step < len(self.Y):
            y_text += f"Y[{self.current_step:2d}] = calculating... [NOW]\n"
        
        # Show pending values (limited to avoid clutter)
        pending_count = min(5, len(self.Y) - self.current_step - 1)
        for i in range(self.current_step + 1, self.current_step + 1 + pending_count):
            if i < len(self.Y):
                y_text += f"Y[{i:2d}] = pending [WAIT]\n"
        
        if len(self.Y) - self.current_step - 1 > 5:
            y_text += f"... and {len(self.Y) - self.current_step - 6} more\n"
        
        self.ax_y_progress.text(0.05, 0.95, y_text, transform=self.ax_y_progress.transAxes,
                              fontsize=9, verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
        
        # Display computation details with better spacing (moved down to avoid covering 2D plot)
        if self.current_computation:
            self.ax_computation.text(0.02, 0.75, "Current Calculation:", fontsize=12, fontweight='bold')
            self.ax_computation.text(0.02, 0.45, self.current_computation, fontsize=10, 
                                   fontfamily='monospace',
                                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
            
            # Show kernel values
            kernel_text = f"Original Kernel H: [{', '.join([f'{h:.2f}' for h in self.H_original])}]"
            self.ax_computation.text(0.02, 0.15, kernel_text, fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Update main title with better positioning
        self.fig.suptitle(f'1D Convolution Visualization - Step {self.current_step + 1} of {self.max_steps}', 
                         fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout with more space for title and new components
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98, hspace=0.4, wspace=0.15)
        self.fig.canvas.draw()
        
    def next_step(self, event=None):
        """Advance to next step in convolution."""
        if self.current_step < self.max_steps:
            # Compute and store the convolution result
            result, x_slice, h_flipped = self.compute_convolution_step()
            self.Y[self.current_step] = result
            self.current_step += 1
            self.update_display()
        else:
            print("Convolution complete! Click Reset to start over.")
            
    def reset(self, event=None):
        """Reset the visualization to initial state with same vectors."""
        self.current_step = 0
        self.Y = np.zeros(self.output_size)
        self.current_computation = ""
        self.update_display()
        
    def new_random(self, event=None):
        """Generate new random vectors and reset."""
        # Generate new random values
        self.X = np.random.uniform(-2, 2, self.x_size)
        self.H_original = np.random.uniform(-1, 1, self.kernel_size)
        self.reset()
        
    def auto_play(self, event=None):
        """Automatically play through all steps."""
        import time
        original_step = self.current_step
        while self.current_step < self.max_steps:
            self.next_step()
            plt.pause(1.5)  # Wait 1.5 seconds between steps
    
    def show_correlation(self, event=None):
        """Launch the correlation visualization window."""
        import subprocess
        import sys
        try:
            # Launch correlation demo as separate process
            subprocess.Popen([sys.executable, 'correlation_demo.py'])
            print("Launching Correlation Demo in separate window...")
        except Exception as e:
            print(f"Could not launch correlation demo: {e}")
            print("Please run 'python correlation_demo.py' manually.")
            
    def show(self):
        """Display the visualization."""
        plt.show()

def main():
    """Main function to run the enhanced convolution visualizer."""
    print("=" * 70)
    print("Enhanced 1D Convolution & Correlation Visualization")
    print("=" * 70)
    print("\nThis program demonstrates 1D convolution step by step:")
    print("• Vector X: Input signal (length 10)")
    print("• Kernel H: Convolution kernel (length 5)")
    print("• Output Y: Full convolution result (length 14)")
    print("• 2D Plot: Vector representation using H[0] and H[1]")
    print("• Sliding Window: Shows kernel moving over padded input")
    print("\nMain Controls:")
    print("• Next Step: Advance one convolution step")
    print("• Reset: Restart with same vectors")
    print("• Auto Play: Run through all steps automatically")
    print("• New Random: Generate new random vectors")
    print("• Correlation Demo: Open 2D correlation visualization")
    print("\nCorrelation Demo Features:")
    print("• Interactive slider from -1 to +1 correlation")
    print("• Real-time scatter plot updates")
    print("• Shows circular (r=0) to linear (r=±1) patterns")
    print("• Educational information panel")
    print("\nMathematical Details:")
    print("• Full convolution: output_size = input_size + kernel_size - 1")
    print("• Zero padding is applied to input for boundary handling")
    print("• Kernel is flipped for true convolution (not correlation)")
    print("• Each step shows the detailed dot product calculation")
    print("=" * 70)
    
    visualizer = EnhancedConvolutionVisualizer()
    visualizer.show()

if __name__ == "__main__":
    main()
