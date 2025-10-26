import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import os
import requests

# SET PAGE CONFIG FIRST 
st.set_page_config(page_title="IntelliScan", layout="centered")

# Load model directly in Streamlit
@st.cache_resource
def load_model():
    model_path = "intelliscan_model_final.pkl"
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Dropbox...")
        try:
            import requests
            
            # Get share link from Dropbox and change ?dl=0 to ?dl=1
            dropbox_url = "https://www.dropbox.com/scl/fi/8xlpfbuqcwtrg7pzkomw2/intelliscan_model_final.pkl?rlkey=4zxniwqakng5hwoyw0d7kxgvx&st=qj6jw70n&dl=1"
            
            response = requests.get(dropbox_url, stream=True, timeout=30)
            
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(model_path)
                # st.success(f"✅ Model downloaded! ({file_size//1000000}MB)")
            else:
                st.error("❌ Model download failed")
                return None
                
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            return None
    
    try:
        model = load_learner(model_path)
        # st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


# Load model
model = load_model()

# ✅ NOW the rest of your UI
st.title("📄 IntelliScan - AI Document Classifier")

st.sidebar.markdown("---")
st.sidebar.info("""
**Note for reviewers:**
This is a free-tier deployment that sleeps after inactivity.

**If the app is loading slowly:**
- Wait 30-60 seconds for wake-up
- Refresh the page  
- First load might take 2-3 minutes
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 For Employers")
st.sidebar.markdown("""
**Technical Highlights:**
- FastAI transfer learning
- Model deployment pipeline  
- Docker containerization
- Cloud deployment (Render)
- Production error handling
- Batch processing capabilities
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Quick Test")
st.sidebar.markdown("""
Try uploading:
- Invoice images
- Receipt photos  
- Contract scans
- Research papers
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Info")
st.sidebar.markdown("""
- **Accuracy:** 95%+
- **Classes:** 4 document types
- **Framework:** FastAI + PyTorch
- **Support:** JPG, JPEG, PNG
""")

if model is None:
    st.error("🚫 Model not available. Please check the logs above.")
    st.info("The model file might be corrupted or download failed.")
else:
    st.write("Upload documents to classify as Invoice, Receipt, Contract, or Research Paper")
    
    # Single file prediction function
    def predict_single_image(image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            temp_path = "temp_prediction.jpg"
            image.save(temp_path, "JPEG")
            prediction, _, probs = model.predict(temp_path)
            confidence = probs.max().item()
            class_probs = {model.dls.vocab[i]: float(probs[i]) for i in range(len(model.dls.vocab))}
            os.remove(temp_path)
            
            return {
                "document_type": str(prediction),
                "confidence": confidence,
                "all_probabilities": class_probs,
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Single File", "📚 Batch Processing"])

    with tab1:
        # Single File Processing
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="single")

        if uploaded_file is not None:
            # Display preview
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_column_width=True)

            if st.button('Classify Document', type="primary", key="single_btn"):
                with st.spinner('Analyzing with AI...'):
                    result = predict_single_image(image)
                    
                    if result["status"] == "success":
                        st.success("✅ Classification Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Document Type", result['document_type'].title())
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        st.subheader("Detailed Analysis:")
                        for doc_type, prob in result['all_probabilities'].items():
                            st.progress(prob, text=f'{doc_type.title()}: {prob:.2%}')
                    else:
                        st.error(f"❌ Classification failed: {result['error']}")

    with tab2:
        # Batch processing
        st.subheader("Process Multiple Files")
        uploaded_files = st.file_uploader("Choose multiple files", type=["jpg", "jpeg", "png"], 
                                         accept_multiple_files=True, key="batch")

        if uploaded_files:
            st.write(f"📁 Selected {len(uploaded_files)} files")

            if st.button("🔍 Classify All Documents", type="primary", key="batch_btn"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                        
                        # Process each file
                        image = Image.open(file)
                        result = predict_single_image(image)
                        
                        if result["status"] == "success":
                            results.append({
                                "filename": file.name,
                                "document_type": result['document_type'],
                                "confidence": f"{result['confidence']:.2%}",
                                "status": "✅ Success"
                            })
                        else:
                            results.append({
                                "filename": file.name,
                                "document_type": "Failed",
                                "confidence": "0%",
                                "status": f"❌ {result['error']}"
                            })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        results.append({
                            "filename": file.name,
                            "document_type": "Failed",
                            "confidence": "0%",
                            "status": f"❌ {str(e)}"
                        })

                # Display results
                if results:
                    df = pd.DataFrame(results)
                    st.subheader("📊 Results Summary")
                    st.dataframe(df)
                    
                    # Export to CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="intelliscan_results.csv",
                        mime="text/csv"
                    )

# Footer
# st.markdown("---")
# st.markdown("**Model Accuracy:** 95%+ | **Supported Types:** Invoices, Receipts, Contracts, Research Papers | **Features:** Single & Batch Processing")




