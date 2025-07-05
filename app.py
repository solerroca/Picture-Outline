import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Image as ReportLabImage
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from skimage import filters, feature, morphology, color
from scipy import ndimage
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Image to Outline Converter",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    .processing-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .upload-section {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 2px dashed #4CAF50;
    }
    
    .download-section {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 2px solid #ff9800;
    }
    
    .image-container {
        text-align: center;
        margin: 1rem 0;
    }
    
    .comparison-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
    }
    
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .info-message {
        background: #cce7ff;
        color: #004085;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
    

</style>
""", unsafe_allow_html=True)

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
    
    def load_image(self, image_file):
        """Load and process uploaded image"""
        try:
            # Read image
            image = Image.open(image_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            self.original_image = np.array(image)
            
            return True
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return False
    
    def create_outline(self, method='canny', blur_kernel=5, threshold1=50, threshold2=150, 
                      line_thickness=1, invert=False):
        """Create outline from image using specified method"""
        if self.original_image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Apply edge detection based on method
        if method == 'canny':
            edges = cv2.Canny(blurred, threshold1, threshold2)
        elif method == 'sobel':
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method == 'laplacian':
            edges = cv2.Laplacian(blurred, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        elif method == 'adaptive':
            edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            edges = 255 - edges  # Invert for edges
        
        # Apply morphological operations to clean up
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Invert if requested (black lines on white background)
        if invert:
            edges = 255 - edges
        
        self.processed_image = edges
        return edges
    
    def resize_for_print(self, size_option, custom_width=None, custom_height=None, dpi=300):
        """Resize image for printing"""
        if self.processed_image is None:
            return None
        
        # Define sizes in inches
        sizes = {
            'Letter': (8.5, 11),
            'A4': (8.27, 11.69),
            'Legal': (8.5, 14),
            'A3': (11.69, 16.54),
            'Custom': (custom_width/25.4 if custom_width else 8.5, 
                      custom_height/25.4 if custom_height else 11)
        }
        
        if size_option not in sizes:
            size_option = 'Letter'
        
        target_width, target_height = sizes[size_option]
        
        # Calculate target dimensions in pixels
        target_width_px = int(target_width * dpi)
        target_height_px = int(target_height * dpi)
        
        # Resize image while maintaining aspect ratio
        h, w = self.processed_image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > target_width_px / target_height_px:
            # Image is wider, fit to width
            new_width = target_width_px
            new_height = int(new_width / aspect_ratio)
        else:
            # Image is taller, fit to height
            new_height = target_height_px
            new_width = int(new_height * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(self.processed_image, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        
        return resized

def create_pdf(image_array, size_option='Letter', filename='outline.pdf'):
    """Create PDF from image array"""
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    # Define page sizes
    page_sizes = {
        'Letter': letter,
        'A4': A4,
        'Legal': (8.5*inch, 14*inch),
        'A3': (11.69*inch, 16.54*inch),
        'Custom': letter  # Default fallback
    }
    
    page_size = page_sizes.get(size_option, letter)
    
    # Create PDF
    doc = SimpleDocTemplate(temp_path, pagesize=page_size,
                          rightMargin=0.5*inch, leftMargin=0.5*inch,
                          topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Save PIL image to temporary file
    img_temp_path = os.path.join(temp_dir, 'temp_image.png')
    pil_image.save(img_temp_path)
    
    # Calculate image size to fit page
    page_width, page_height = page_size
    available_width = page_width - 1*inch
    available_height = page_height - 1*inch
    
    # Get image dimensions
    img_width, img_height = pil_image.size
    aspect_ratio = img_width / img_height
    
    # Calculate fitted dimensions
    if aspect_ratio > available_width / available_height:
        # Image is wider, fit to width
        fitted_width = available_width
        fitted_height = fitted_width / aspect_ratio
    else:
        # Image is taller, fit to height
        fitted_height = available_height
        fitted_width = fitted_height * aspect_ratio
    
    # Create ReportLab image
    img = ReportLabImage(img_temp_path, width=fitted_width, height=fitted_height)
    
    # Build PDF
    story = [img]
    doc.build(story)
    
    # Clean up temporary image
    os.unlink(img_temp_path)
    
    return temp_path



def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Image to Outline Converter</h1>
        <p>Transform any image into a clean black-and-white outline perfect for printing and watercolor painting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = ImageProcessor()
    
    # Sidebar with upload and settings
    st.sidebar.header("🎛️ Upload & Settings")
    
    # Upload section in sidebar
    st.sidebar.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.sidebar.subheader("📤 Upload Image")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Drag and drop an image file here, or click to browse"
    )
    
    # Separator
    st.sidebar.markdown("**OR**")
    
    # Direct clipboard paste section
    st.sidebar.markdown("**📋 Paste Screenshot Directly:**")
    
    # Instructions for direct pasting
    st.sidebar.markdown("""
    <div style="background-color: #e8f5e8; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; font-size: 0.8rem;">
        <strong>How to paste screenshots:</strong><br/>
        1️⃣ Take a screenshot (Cmd+Shift+4 on Mac, Windows+Shift+S on PC)<br/>
        2️⃣ Paste the base64 data below<br/>
        3️⃣ The image will be processed automatically<br/>
    </div>
    """, unsafe_allow_html=True)

    # Fallback text area for base64 input
    base64_input = st.sidebar.text_area(
        "Paste base64 image data here:",
        height=100,
        placeholder="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        help="You can paste base64 encoded image data here. Take a screenshot, convert to base64, and paste."
    )
    
    # Process base64 input if provided
    if base64_input and base64_input.strip():
        if base64_input.startswith("data:image"):
            try:
                # Extract the actual base64 data
                base64_data = base64_input.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                img_file = io.BytesIO(img_bytes)
                
                # Reset state
                st.session_state.processor.original_image = None
                st.session_state.processor.processed_image = None
                st.session_state.last_method = None
                st.session_state.last_blur_kernel = None
                st.session_state.last_threshold1 = None
                st.session_state.last_threshold2 = None
                st.session_state.last_line_thickness = None
                st.session_state.last_invert = None
                st.session_state.current_file_name = "pasted_screenshot"

                if st.session_state.processor.load_image(img_file):
                    st.sidebar.success("✅ Image loaded from base64 data!")
                else:
                    st.sidebar.error("❌ Failed to load image from base64 data")
            except Exception as ex:
                st.sidebar.error(f"❌ Error decoding base64 image: {ex}")
        else:
            st.sidebar.warning("⚠️ Please paste valid base64 image data starting with 'data:image'")
    
    # Additional helper for users
    if st.sidebar.button("ℹ️ How to get base64 from screenshot"):
        st.sidebar.markdown("""
        <div style="background-color: #f0f8f0; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; font-size: 0.8rem;">
            <strong>To convert screenshot to base64:</strong><br/>
            1. Take a screenshot<br/>
            2. Go to <a href="https://www.base64-image.de/" target="_blank">base64-image.de</a><br/>
            3. Upload your screenshot<br/>
            4. Copy the generated base64 string<br/>
            5. Paste it in the text area above
        </div>
        """, unsafe_allow_html=True)
    
    # Track current uploaded file to detect changes
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    
    # Image loading
    if uploaded_file is not None:
        # Check if this is a new file (different from previous)
        new_file_uploaded = (
            st.session_state.current_file_name != uploaded_file.name or
            st.session_state.processor.original_image is None
        )
        
        if new_file_uploaded:
            # Clear previous images and reset processor
            st.session_state.processor.original_image = None
            st.session_state.processor.processed_image = None
            
            # Reset processing parameters to force regeneration
            st.session_state.last_method = None
            st.session_state.last_blur_kernel = None
            st.session_state.last_threshold1 = None
            st.session_state.last_threshold2 = None
            st.session_state.last_line_thickness = None
            st.session_state.last_invert = None
            
            # Update current file name
            st.session_state.current_file_name = uploaded_file.name
        
        if st.session_state.processor.load_image(uploaded_file):
            if new_file_uploaded:
                st.sidebar.success("✅ New image loaded successfully!")
            
            # Image info in sidebar
            h, w, c = st.session_state.processor.original_image.shape
            st.sidebar.info(f"📏 Image dimensions: {w} x {h} pixels")
            
            # Initialize processing parameters in session state if not present
            if 'last_method' not in st.session_state:
                st.session_state.last_method = None
            if 'last_blur_kernel' not in st.session_state:
                st.session_state.last_blur_kernel = None
            if 'last_threshold1' not in st.session_state:
                st.session_state.last_threshold1 = None
            if 'last_threshold2' not in st.session_state:
                st.session_state.last_threshold2 = None
            if 'last_line_thickness' not in st.session_state:
                st.session_state.last_line_thickness = None
            if 'last_invert' not in st.session_state:
                st.session_state.last_invert = None
    else:
        # No file uploaded, clear current file name
        if st.session_state.current_file_name is not None:
            st.session_state.current_file_name = None
            st.session_state.processor.original_image = None
            st.session_state.processor.processed_image = None
    
    # Print Settings in sidebar - only show when outline is generated
    if st.session_state.processor.processed_image is not None:
        st.sidebar.markdown('<div class="processing-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📏 Print Settings")
        
        # Size options
        size_option = st.sidebar.selectbox(
            "Output Size",
            ['Letter', 'A4', 'Legal', 'A3', 'Custom'],
            help="Choose the output size for printing"
        )
        
        if size_option == 'Custom':
            custom_width = st.sidebar.number_input("Width (mm)", min_value=50, max_value=500, value=210)
            custom_height = st.sidebar.number_input("Height (mm)", min_value=50, max_value=500, value=297)
        else:
            custom_width = custom_height = None
        
        dpi = st.sidebar.selectbox("Print Quality (DPI)", [150, 300, 600], index=1)
        
        # Resize for print
        print_image = st.session_state.processor.resize_for_print(
            size_option, custom_width, custom_height, dpi
        )
        
        if print_image is not None:
            st.sidebar.info(f"📐 Print size: {print_image.shape[1]} x {print_image.shape[0]} pixels")
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Download section in sidebar
        st.sidebar.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.sidebar.subheader("💾 Download Options")
        
        # PNG download
        if st.sidebar.button("📥 Download PNG", type="secondary"):
            img_to_download = print_image if print_image is not None else st.session_state.processor.processed_image
            pil_image = Image.fromarray(img_to_download)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            st.sidebar.download_button(
                label="💾 Download PNG File",
                data=img_bytes.getvalue(),
                file_name=f"outline_{size_option.lower()}.png",
                mime="image/png"
            )
        
        # PDF download
        if st.sidebar.button("📄 Download PDF", type="secondary"):
            img_to_download = print_image if print_image is not None else st.session_state.processor.processed_image
            
            with st.spinner("Creating PDF..."):
                pdf_path = create_pdf(img_to_download, size_option)
                
                with open(pdf_path, 'rb') as pdf_file:
                    st.sidebar.download_button(
                        label="📄 Download PDF File",
                        data=pdf_file.read(),
                        file_name=f"outline_{size_option.lower()}.pdf",
                        mime="application/pdf"
                    )
                
                # Clean up
                os.unlink(pdf_path)
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.processor.original_image is not None:
        # Processing options section on main page
        st.markdown('<div class="processing-section">', unsafe_allow_html=True)
        st.subheader("⚙️ Processing Options")
        st.markdown("*Outline updates automatically as you adjust the settings below*")
        
        # Add a subtle indicator that auto-processing is active
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem; border-left: 4px solid #4CAF50;">
            <small>🔄 Auto-processing enabled - Changes apply instantly</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for better layout of processing options
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Processing parameters
            method = st.selectbox(
                "Edge Detection Method",
                ['canny', 'sobel', 'laplacian', 'adaptive'],
                help="Choose the edge detection algorithm"
            )
            
            blur_kernel = st.slider("Blur Kernel", 1, 15, 5, step=2,
                                  help="Higher values create smoother outlines")
        
        with col2:
            threshold1 = st.slider("Threshold 1", 10, 200, 50,
                                 help="Lower edge detection threshold")
            threshold2 = st.slider("Threshold 2", 50, 300, 150,
                                 help="Upper edge detection threshold")
        
        with col3:
            line_thickness = st.slider("Line Thickness", 1, 5, 3,
                                     help="Make outline lines thicker")
            invert = st.checkbox("Invert (Black lines on white)", value=True,
                               help="Recommended for printing")
        
        # Check if parameters have changed or if this is the first time processing
        parameters_changed = (
            st.session_state.last_method != method or
            st.session_state.last_blur_kernel != blur_kernel or
            st.session_state.last_threshold1 != threshold1 or
            st.session_state.last_threshold2 != threshold2 or
            st.session_state.last_line_thickness != line_thickness or
            st.session_state.last_invert != invert or
            st.session_state.processor.processed_image is None
        )
        
        # Auto-generate outline if parameters changed
        if parameters_changed:
            with st.spinner("🔄 Generating outline..."):
                outline = st.session_state.processor.create_outline(
                    method=method,
                    blur_kernel=blur_kernel,
                    threshold1=threshold1,
                    threshold2=threshold2,
                    line_thickness=line_thickness,
                    invert=invert
                )
                
                # Update session state with current parameters
                st.session_state.last_method = method
                st.session_state.last_blur_kernel = blur_kernel
                st.session_state.last_threshold1 = threshold1
                st.session_state.last_threshold2 = threshold2
                st.session_state.last_line_thickness = line_thickness
                st.session_state.last_invert = invert
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display images side by side
        if st.session_state.processor.processed_image is not None:
            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
            st.subheader("📸 Original vs Outline")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(st.session_state.processor.original_image, 
                        use_container_width=True)
            
            with col2:
                st.markdown("**Generated Outline**")
                st.image(st.session_state.processor.processed_image, 
                        use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display processing information
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Method", method.capitalize())
            with col2:
                st.metric("Blur Kernel", f"{blur_kernel}px")
            with col3:
                st.metric("Line Thickness", f"{line_thickness}px")
            with col4:
                st.metric("Inverted", "Yes" if invert else "No")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No image uploaded
        st.markdown("""
        <div class="info-message">
            <h3>👋 Welcome to Image to Outline Converter!</h3>
            <p>Upload an image using the sidebar to get started. Your image will be converted to a clean outline perfect for:</p>
            <ul>
                <li>🎨 Watercolor painting</li>
                <li>🖍️ Coloring pages</li>
                <li>📝 Sketching practice</li>
                <li>🖨️ Print-ready artwork</li>
            </ul>
            <p><strong>Supported formats:</strong> PNG, JPG, JPEG, BMP, TIFF</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 