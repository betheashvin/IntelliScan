import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import os
import requests

# ✅ SET PAGE CONFIG FIRST - ONLY ONCE
st.set_page_config(page_title="IntelliScan", layout="centered")

# Load model directly in Streamlit
@st.cache_resource
def load_model():
    model_path = "intelliscan_model_final.pkl"
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive...")
        try:
            import requests
            import re
            
            file_id = "1HGduInluShD47GUvVYUeDXKqHyjhP0wv"
            session = requests.Session()
            
            # First, get the virus scan warning page
            URL = "https://drive.google.com/uc"
            response = session.get(URL, params={'id': file_id, 'export': 'download'}, stream=True)
            
            # Extract the confirmation token from the HTML
            content = response.text
            token = None
            
            # Look for the confirmation token in the HTML
            match = re.search(r"confirm=([0-9A-Za-z_]+)", content)
            if match:
                token = match.group(1)
            else:
                # Alternative pattern
                match = re.search(r'<input type="hidden" name="confirm" value="([0-9A-Za-z_]+)"', content)
                if match:
                    token = match.group(1)
            
            if token:
                st.write(f"🔐 Found confirmation token: {token}")
                
                # Download with confirmation token
                params = {'id': file_id, 'confirm': token, 'export': 'download'}
                response = session.get(URL, params=params, stream=True)
                
                # Download the actual file
                total_size = int(response.headers.get('content-length', 0))
                st.write(f"📦 Actual file size: {total_size} bytes")
                
                if total_size > 10000000:  # If it's a reasonable size
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(model_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = downloaded / total_size
                                    progress_bar.progress(min(progress, 1.0))
                                    status_text.text(f"Downloaded: {downloaded}/{total_size} bytes ({progress:.1%})")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Verify final file size
                    final_size = os.path.getsize(model_path)
                    st.write(f"✅ Final file size: {final_size} bytes")
                    
                    if final_size > 10000000:
                        st.success("🎉 Model downloaded successfully!")
                    else:
                        st.error(f"❌ File still too small: {final_size} bytes")
                        os.remove(model_path)
                        return None
                else:
                    st.error(f"❌ File size too small: {total_size} bytes")
                    return None
            else:
                st.error("❌ Could not extract confirmation token from Google Drive")
                st.info("💡 Try downloading the file manually and uploading it:")
                
                # Fallback: Manual upload
                uploaded_model = st.file_uploader("Upload model file manually", type=['pkl'], key="model_upload")
                if uploaded_model is not None:
                    with open(model_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    st.success("✅ Model uploaded manually!")
                    st.rerun()
                return None
                
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            import traceback
            st.write("Detailed error:", traceback.format_exc())
            return None
    
    # Load model
    try:
        model = load_learner(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return None


# Load model
model = load_model()

# ✅ NOW the rest of your UI
st.title("📄 IntelliScan - AI Document Classifier")

if model is None:
    st.error("🚫 Model not available. Please check the logs above.")
    st.info("The model file might be corrupted or download failed.")
else:
    st.write("Upload documents to classify as Invoice, Receipt, Contract, or Research Paper")
    
    # Single file prediction function
    def predict_single_image(image):
        try:
            temp_path = "temp_prediction.jpg"
            image.save(temp_path)
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
st.markdown("---")
st.markdown("**Model Accuracy:** 99.4% | **Supported Types:** Invoices, Receipts, Contracts, Research Papers | **Features:** Single & Batch Processing")


