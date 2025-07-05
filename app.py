import streamlit as st
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

# Page config
st.set_page_config(
    page_title="Image to Outline Converter",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
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
    
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    
    .processing-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .download-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #d4edda;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
        }
        
        .upload-section {
            padding: 1rem;
        }
        
        .processing-section {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class ImageProcessor:
    """Handle all image processing operations"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        
    def load_image(self, image_file):
        """Load image from uploaded file"""
        try:
            image = Image.open(image_file)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            self.original_image = np.array(image)
            return True
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return False
    
    def create_outline(self, method='canny', blur_kernel=5, threshold1=50, threshold2=150, 
                      line_thickness=1, invert=False):
        """Create outline from image using various edge detection methods"""
        if self.original_image is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        if method == 'canny':
            # Canny edge detection
            edges = cv2.Canny(blurred, threshold1, threshold2)
            
        elif method == 'sobel':
            # Sobel edge detection
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.uint8(edges / edges.max() * 255)
            
        elif method == 'laplacian':
            # Laplacian edge detection
            edges = cv2.Laplacian(blurred, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            
        elif method == 'adaptive':
            # Adaptive threshold for artistic effect
            edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 10)
        
        # Dilate edges to make them thicker if needed
        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Invert if requested (black lines on white background)
        if invert:
            edges = 255 - edges
            
        self.processed_image = edges
        return edges
    
    def resize_for_print(self, size_option, custom_width=None, custom_height=None, dpi=300):
        """Resize image for printing at specified size"""
        if self.processed_image is None:
            return None
            
        # Define standard sizes in inches
        sizes = {
            'A4': (8.27, 11.69),
            'Letter': (8.5, 11.0),
            'Legal': (8.5, 14.0),
            'A3': (11.69, 16.54),
            'Custom': (custom_width/25.4 if custom_width else 8.5, 
                      custom_height/25.4 if custom_height else 11.0)
        }
        
        target_size = sizes.get(size_option, sizes['Letter'])
        
        # Calculate target dimensions in pixels
        target_width_px = int(target_size[0] * dpi)
        target_height_px = int(target_size[1] * dpi)
        
        # Resize while maintaining aspect ratio
        h, w = self.processed_image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > target_size[0] / target_size[1]:
            # Width is the limiting factor
            new_width = target_width_px
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = target_height_px
            new_width = int(new_height * aspect_ratio)
        
        resized = cv2.resize(self.processed_image, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        
        return resized

def create_pdf(image_array, size_option='Letter', filename='outline.pdf'):
    """Create PDF from image array"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
    
    # Define page sizes
    page_sizes = {
        'A4': A4,
        'Letter': letter,
        'Legal': (8.5*inch, 14*inch),
        'A3': (11.69*inch, 16.54*inch)
    }
    
    page_size = page_sizes.get(size_option, letter)
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=page_size)
    
    # Convert image to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_tmp:
        pil_image.save(img_tmp.name)
        
        # Add image to PDF
        story = []
        img = ReportLabImage(img_tmp.name)
        
        # Scale image to fit page
        img_width, img_height = pil_image.size
        page_width, page_height = page_size
        
        # Calculate scaling factor
        scale_x = (page_width - 2*inch) / img_width
        scale_y = (page_height - 2*inch) / img_height
        scale = min(scale_x, scale_y)
        
        img.drawWidth = img_width * scale
        img.drawHeight = img_height * scale
        
        story.append(img)
        doc.build(story)
    
    # Clean up temp image file
    os.unlink(img_tmp.name)
    
    return pdf_path

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
        "Choose an image file or screenshot",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Drag and drop an image file or saved screenshot here, or click to browse"
    )
    
    # Separator
    st.sidebar.markdown("**OR**")
    
    # Screenshot/Clipboard instructions
    st.sidebar.markdown("**📋 Use Screenshots:**")
    
    # Improved instructions for screenshots
    st.sidebar.markdown("""
    <div style="background-color: #f0f4f8; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; font-size: 0.8rem;">
        <strong>How to use screenshots:</strong><br/>
        1️⃣ Take a screenshot (Cmd+Shift+4 on Mac, Windows+Shift+S on PC)<br/>
        2️⃣ <strong>Save the screenshot as a file</strong> (PNG/JPG)<br/>
        3️⃣ Use the file uploader above to upload the saved screenshot<br/><br/>
        💡 <em>Direct paste from clipboard isn't supported in web browsers for security reasons</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick screenshot guide
    st.sidebar.markdown("**🖱️ Quick Screenshot Guide:**")
    st.sidebar.markdown("""
    <div style="border: 2px dashed #4CAF50; padding: 1rem; text-align: center; border-radius: 8px; background-color: #f8fff8; font-size: 0.8rem;">
        📸 <strong>Screenshot Shortcuts:</strong><br/>
        🍎 <strong>Mac:</strong> Cmd + Shift + 4 (select area)<br/>
        🪟 <strong>Windows:</strong> Windows + Shift + S<br/>
        🐧 <strong>Linux:</strong> PrtSc or Shift + PrtSc<br/><br/>
        Screenshots auto-save to Desktop/Downloads folder<br/>
        Then drag the file to the uploader above! 📤
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
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
                
                if outline is not None:
                    # Update session state with current parameters
                    st.session_state.last_method = method
                    st.session_state.last_blur_kernel = blur_kernel
                    st.session_state.last_threshold1 = threshold1
                    st.session_state.last_threshold2 = threshold2
                    st.session_state.last_line_thickness = line_thickness
                    st.session_state.last_invert = invert
                else:
                    st.error("❌ Error generating outline")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Side-by-side image comparison
        st.subheader("🖼️ Image Comparison")
        
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Original Image")
            st.image(st.session_state.processor.original_image, 
                    caption="Original Image", use_container_width=True)
        
        with col2:
            if st.session_state.processor.processed_image is not None:
                st.markdown("### Generated Outline")
                st.image(st.session_state.processor.processed_image, 
                        caption="Generated Outline", use_container_width=True)
            else:
                st.markdown("### Generated Outline")
                st.info("👆 Adjust the processing options above to generate the outline")
    
    else:
        # Show upload instructions when no image is loaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; color: #666;">
            <h3>👈 Upload an image using the sidebar to get started</h3>
            <p>Supported formats: PNG, JPG, JPEG, BMP, TIFF</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎨 Perfect for watercolor painting, coloring pages, and artistic projects</p>
        <p>💡 Tip: Try different edge detection methods for various artistic effects</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 