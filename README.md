# 🎨 Image to Outline Converter

A user-friendly web application that transforms any image into clean black-and-white outlines, perfect for printing and watercolor painting.

## ✨ Features

- **🔄 Multiple Edge Detection Methods**: Canny, Sobel, Laplacian, and Adaptive algorithms
- **📤 Flexible Image Upload**: Drag-and-drop, file selection, or clipboard paste
- **⚙️ Customizable Processing**: Adjust blur, thresholds, and line thickness
- **📏 Print Optimization**: Support for A4, Letter, Legal, A3, and custom sizes
- **💾 Multiple Export Formats**: Download as PNG or PDF
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🎯 Optimized for Artists**: Clean outlines perfect for watercolor painting and coloring

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Proj_3_Outline_WC
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   The app will automatically open in your default browser at `http://localhost:8501`

## 🎯 Usage Guide

### Step 1: Upload Image
- **File Upload**: Click "Browse files" or drag and drop an image
- **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF
- **Size Limit**: Up to 200MB per file

### Step 2: Adjust Settings
- **Edge Detection Method**: Choose from 4 different algorithms
  - **Canny**: Best for most photos (recommended)
  - **Sobel**: Good for gradient-based edges
  - **Laplacian**: Emphasizes fine details
  - **Adaptive**: Artistic effect with varying thresholds

- **Processing Parameters**:
  - **Blur Kernel**: Reduces noise (1-15)
  - **Threshold 1**: Lower edge threshold (10-200)
  - **Threshold 2**: Upper edge threshold (50-300)
  - **Line Thickness**: Makes lines thicker (1-5)
  - **Invert**: Black lines on white background (recommended for printing)

### Step 3: Generate Outline
- Click "🔄 Generate Outline" to process the image
- View the result in real-time
- Adjust parameters and regenerate as needed

### Step 4: Print Settings
- **Output Size**: Choose from standard paper sizes or custom dimensions
- **Print Quality**: Select DPI (150, 300, or 600)
- **Preview**: See exact print dimensions

### Step 5: Download
- **PNG**: High-quality raster image
- **PDF**: Print-ready document with proper scaling

## 🛠️ Technical Details

### Image Processing Pipeline
1. **Image Loading**: Convert to RGB, handle various formats
2. **Grayscale Conversion**: Optimize for edge detection
3. **Gaussian Blur**: Reduce noise while preserving edges
4. **Edge Detection**: Apply selected algorithm
5. **Morphological Operations**: Adjust line thickness
6. **Inversion**: Convert for optimal printing
7. **Resize**: Scale for chosen paper size while maintaining aspect ratio

### Supported Paper Sizes
- **Letter**: 8.5" × 11" (US standard)
- **A4**: 210mm × 297mm (International standard)
- **Legal**: 8.5" × 14" (US legal)
- **A3**: 297mm × 420mm (Large format)
- **Custom**: User-defined dimensions

### Performance Optimizations
- **Efficient Memory Usage**: Process large images without overwhelming system
- **Responsive Processing**: Real-time parameter adjustment
- **Optimized Algorithms**: Fast edge detection with quality results
- **Smart Resizing**: Maintain aspect ratio while fitting paper sizes

## 📱 Responsive Design

The application is optimized for all device types:
- **Desktop**: Full-featured interface with side-by-side layout
- **Tablet**: Adaptive layout with touch-friendly controls
- **Mobile**: Stacked layout optimized for small screens

## 🎨 Perfect for Artists

### Watercolor Painting
- Clean, single-pixel lines
- Optimal contrast for easy tracing
- Print-ready sizing for standard paper

### Coloring Pages
- Clear boundaries for coloring
- Customizable line thickness
- High-contrast output

### Art Projects
- Multiple artistic styles via different algorithms
- Adjustable detail levels
- Professional print quality

## 📋 Requirements

### Python Packages
- `streamlit>=1.32.0` - Web application framework
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=10.0.0` - Image handling
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization
- `reportlab>=4.0.0` - PDF generation
- `fpdf2>=2.7.0` - Alternative PDF creation
- `scikit-image>=0.21.0` - Advanced image processing
- `scipy>=1.11.0` - Scientific computing
- `plotly>=5.17.0` - Interactive visualizations

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Browser**: Modern browser with JavaScript enabled

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

#### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Heroku
```bash
heroku create your-app-name
heroku config:set STREAMLIT_SERVER_HEADLESS=true
git push heroku main
```

## 🔧 Configuration

### Streamlit Configuration
The app includes optimized settings in `.streamlit/config.toml`:
- Maximum upload size: 200MB
- Custom theme colors
- Performance optimizations

### Environment Variables
```bash
# Optional: Set custom configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🎯 Best Practices

### For Best Results
1. **Image Quality**: Use high-resolution images for better edge detection
2. **Contrast**: Images with good contrast produce cleaner outlines
3. **Simplicity**: Less complex images often yield better results
4. **Experimentation**: Try different methods and parameters

### Troubleshooting
- **Large Files**: Ensure sufficient system memory
- **Slow Processing**: Reduce image size or lower DPI
- **Poor Quality**: Adjust thresholds and try different methods

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

## 🔄 Version History

### v1.0.0
- Initial release
- Core edge detection functionality
- Multiple output formats
- Responsive design
- Print optimization

---

**Created with ❤️ for artists and creative enthusiasts** 