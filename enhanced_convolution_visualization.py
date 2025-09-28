import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches

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
        
        # Create grid layout with better proportions
        gs = self.fig.add_gridspec(4, 3, height_ratios=[1.2, 1.2, 1.2, 0.3], 
                                  width_ratios=[1, 1, 1])
        
        # Vector displays (top row)
        self.ax_x = self.fig.add_subplot(gs[0, 0])
        self.ax_h = self.fig.add_subplot(gs[0, 1])
        self.ax_y = self.fig.add_subplot(gs[0, 2])
        
        # 2D vector visualization (second row, spans all columns)
        self.ax_2d = self.fig.add_subplot(gs[1, :])
        
        # Computation details (third row, spans all columns)
        self.ax_computation = self.fig.add_subplot(gs[2, :])
        
        # Control buttons (bottom row)
        self.ax_buttons = self.fig.add_subplot(gs[3, :])
        
        self.setup_vector_plots()
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
        ax_random = plt.axes([0.6, 0.02, 0.15, 0.04])
        self.btn_random = Button(ax_random, 'New Random')
        self.btn_random.on_clicked(self.new_random)
        
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
        self.ax_2d.clear()
        self.ax_computation.clear()
        
        # Reapply settings
        self.setup_vector_plots()
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
        
        # Display computation details with better spacing
        if self.current_computation:
            self.ax_computation.text(0.02, 0.85, "Current Calculation:", fontsize=13, fontweight='bold')
            self.ax_computation.text(0.02, 0.55, self.current_computation, fontsize=11, 
                                   fontfamily='monospace',
                                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
            
            # Show kernel values
            kernel_text = f"Original Kernel H: [{', '.join([f'{h:.2f}' for h in self.H_original])}]"
            self.ax_computation.text(0.02, 0.25, kernel_text, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            # Show Y vector progress on the right side
            y_progress_text = "Y Vector Progress:\n"
            for i in range(min(self.current_step + 1, len(self.Y))):
                y_progress_text += f"Y[{i}] = {self.Y[i]:.3f}\n"
            if self.current_step < len(self.Y) - 1:
                y_progress_text += f"Y[{self.current_step + 1}] = calculating...\n"
                for i in range(self.current_step + 2, len(self.Y)):
                    y_progress_text += f"Y[{i}] = pending\n"
            
            self.ax_computation.text(0.55, 0.85, y_progress_text, fontsize=10, 
                                   verticalalignment='top',
                                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
        
        # Update main title with better positioning
        self.fig.suptitle(f'1D Convolution Visualization - Step {self.current_step + 1} of {self.max_steps}', 
                         fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout with more space for title
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.06, right=0.96, hspace=0.6, wspace=0.25)
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
            
    def show(self):
        """Display the visualization."""
        plt.show()

def main():
    """Main function to run the enhanced convolution visualizer."""
    print("=" * 60)
    print("Enhanced 1D Convolution Visualization")
    print("=" * 60)
    print("\nThis program demonstrates 1D convolution step by step:")
    print("• Vector X: Input signal (length 10)")
    print("• Kernel H: Convolution kernel (length 5)")
    print("• Output Y: Full convolution result (length 14)")
    print("• 2D Plot: Vector representation using H[0] and H[1]")
    print("\nControls:")
    print("• Next Step: Advance one convolution step")
    print("• Reset: Restart with same vectors")
    print("• Auto Play: Run through all steps automatically")
    print("• New Random: Generate new random vectors")
    print("\nMathematical Details:")
    print("• Full convolution: output_size = input_size + kernel_size - 1")
    print("• Zero padding is applied to input for boundary handling")
    print("• Kernel is flipped for true convolution (not correlation)")
    print("• Each step shows the detailed dot product calculation")
    print("=" * 60)
    
    visualizer = EnhancedConvolutionVisualizer()
    visualizer.show()

if __name__ == "__main__":
    main()
