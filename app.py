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
            import gdown
            
            # Method 1: Try different URL formats
            file_id = "1HGduInluShD47GUvVYUeDXKqHyjhP0wv"
            
            # Try multiple URL formats
            urls_to_try = [
                f"https://drive.google.com/uc?id={file_id}",
                f"https://drive.google.com/uc?export=download&id={file_id}",
                f"https://docs.google.com/uc?export=download&id={file_id}"
            ]
            
            success = False
            for i, url in enumerate(urls_to_try):
                st.write(f"🔄 Trying URL {i+1}: {url}")
                try:
                    gdown.download(url, model_path, quiet=False)
                    
                    if os.path.exists(model_path):
                        file_size = os.path.getsize(model_path)
                        st.write(f"📦 File size: {file_size} bytes")
                        
                        # Check if file is HTML (error page)
                        with open(model_path, 'rb') as f:
                            content = f.read(1000)
                            if b'html' in content.lower() or b'error' in content.lower() or file_size < 5000:
                                st.warning(f"❌ URL {i+1} returned HTML/error page ({file_size} bytes)")
                                os.remove(model_path)
                                continue
                            else:
                                st.success(f"✅ URL {i+1} worked! File size: {file_size} bytes")
                                success = True
                                break
                except Exception as e:
                    st.warning(f"❌ URL {i+1} failed: {e}")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    continue
            
            if not success:
                st.error("🚫 All download methods failed. The file might not be publicly accessible.")
                
                # Show what we're getting
                st.info("🔍 Checking what's being returned...")
                import requests
                test_url = f"https://drive.google.com/uc?id={file_id}"
                response = requests.get(test_url, stream=True)
                st.write(f"Response status: {response.status_code}")
                st.write(f"Response headers: {dict(response.headers)}")
                
                # Show first 500 characters of response
                content_preview = response.text[:500] if response.text else "No content"
                st.write(f"Content preview: {content_preview}")
                
                return None

            # Final verification
            file_size = os.path.getsize(model_path)
            if file_size < 10000000:  # Less than 10MB
                st.error(f"❌ File too small ({file_size} bytes) - likely corrupted")
                os.remove(model_path)
                return None
                
            st.success("✅ Model downloaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
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

