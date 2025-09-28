# L8-CorollationVsConvulation

## 📊 Interactive 1D Convolution Visualization

An educational Python program that demonstrates the mathematical process of 1D convolution step-by-step, showing how a kernel slides over an input signal and computes the convolution result interactively.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%2C%20matplotlib-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

### Visual Components
- **Input Vector X**: Original signal (length 10)
- **Kernel H**: Sliding convolution kernel (length 5) 
- **Output Vector Y**: Convolution results (length 14)
- **2D Vector Plot**: Interactive visualization of H[0:2] × X_slice[0:2]
- **Computation Details**: Step-by-step mathematical calculations

### Interactive Controls
- 🔄 **Next Step**: Advance one convolution step
- 🔁 **Reset**: Restart with same vectors
- ▶️ **Auto Play**: Automatic step-through animation
- 🎲 **New Random**: Generate fresh random vectors

### Mathematical Accuracy
- ✅ Full convolution with zero-padding
- ✅ True convolution (kernel flipping)
- ✅ Detailed dot product calculations
- ✅ Real-time 2D vector representation

## 🚀 Quick Start

### Prerequisites
- **Python 3.8 or higher** (3.9-3.12 recommended)
- Windows 10/11
- PowerShell 5.1 or PowerShell Core 7+

#### Check Your Python Version:
```powershell
python --version
# Should show Python 3.8.x or higher
```

### Installation with UV (Recommended)

#### Step 1: Install UV Package Manager
Open PowerShell as Administrator and run:
```powershell
# Install UV using the official installer
Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
```

#### Step 2: Clone the Repository
```powershell
# Clone the repository
git clone https://github.com/rmisegal/L8-CorollationVsConvulation.git

# Navigate to project directory
cd L8-CorollationVsConvulation
```

#### Step 3: Set Up Virtual Environment and Install Dependencies
```powershell
# Create virtual environment and install dependencies
uv venv

# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Install project dependencies
uv pip install -r requirements.txt
```

#### Step 4: Run the Visualization
```powershell
# Run the main program
python enhanced_convolution_visualization.py
```

### Alternative Installation (without UV)

If you prefer using standard pip:
```powershell
# Clone repository
git clone https://github.com/rmisegal/L8-CorollationVsConvulation.git
cd L8-CorollationVsConvulation

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the program
python enhanced_convolution_visualization.py
```

## 📖 How to Use

1. **Launch the Program**: Run the Python script to open the interactive window
2. **Understand the Layout**:
   - Top row: Three vector displays (X, H, Y)
   - Middle: 2D vector visualization
   - Bottom: Computation details and controls
3. **Step Through Convolution**:
   - Click "Next Step" to advance one calculation
   - Watch the kernel slide across the input
   - Observe the 2D vector changes in real-time
4. **Experiment**:
   - Use "New Random" to try different vectors
   - Use "Auto Play" for continuous animation
   - Use "Reset" to restart with same data

## 🔬 Mathematical Background

### Convolution Formula
```
Y[n] = Σ(k=0 to K-1) X[n-k] × H[k]
```

### Key Concepts Demonstrated
- **Zero Padding**: Input extended with zeros for boundary handling
- **Kernel Flipping**: True convolution vs correlation
- **Sliding Window**: How the kernel moves across the input
- **Dot Product**: Mathematical computation at each step
- **2D Representation**: Visualization of vector components

### Output Size Calculation
```
Output Length = Input Length + Kernel Length - 1
             = 10 + 5 - 1 = 14
```

## 📁 Project Structure

```
L8-CorollationVsConvulation/
├── enhanced_convolution_visualization.py  # Main program
├── requirements.txt                       # Dependencies for pip
├── pyproject.toml                        # UV project configuration
├── README.md                             # This file
└── .gitignore                           # Git ignore rules
```

## 🛠️ Troubleshooting

### Common Issues

#### PowerShell Execution Policy Error
```powershell
# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### UV Not Found
```powershell
# Restart PowerShell after UV installation, or add to PATH manually:
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
```

#### Virtual Environment Activation Issues
```powershell
# If activation fails, try:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

#### Display Issues
- Ensure you have a GUI backend for matplotlib
- On Windows, this is usually automatic with standard Python installations
- If issues persist, try: `pip install PyQt5`

### System Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 512MB minimum
- **Display**: GUI support required

## 🎓 Educational Value

This visualization helps students understand:
- **Convolution vs Correlation**: See the difference with kernel flipping
- **Boundary Effects**: How zero-padding affects results
- **Step-by-Step Process**: Mathematical computation breakdown
- **Vector Operations**: 2D representation of component interactions
- **Signal Processing**: Foundation concepts for DSP and ML

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name** - [rmisegal@gmail.com](mailto:rmisegal@gmail.com)

## 🙏 Acknowledgments

- Built with Python, NumPy, and Matplotlib
- Inspired by signal processing and machine learning education needs
- Designed for interactive learning and visualization

---

### 📞 Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Contact: [rmisegal@gmail.com](mailto:rmisegal@gmail.com)

**Happy Learning! 🎉**
