import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import os
import requests
import io

# Load model directly in Streamlit
@st.cache_resource
def load_model():
    model_path = "intelliscan_model_final.pkl"
    
    # Download model if not exists
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive...")
        try:
            file_id = "1HGduInluShD47GUvVYUeDXKqHyjhP0wv"
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = requests.get(download_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None
    
    # Load the model
    try:
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# Load model
model = load_model()

st.set_page_config(page_title="IntelliScan", layout="centered")
st.title("📄 IntelliScan - AI Document Classifier")
st.write("Upload documents to classify as Invoice, Receipt, Contract, or Research Paper")

# Single file prediction function
def predict_single_image(image):
    try:
        # Save temp file
        temp_path = "temp_prediction.jpg"
        image.save(temp_path)
        
        # Predict
        prediction, _, probs = model.predict(temp_path)
        confidence = probs.max().item()
        
        # Get all probabilities
        class_probs = {model.dls.vocab[i]: float(probs[i]) for i in range(len(model.dls.vocab))}
        
        # Clean up
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
            if model is None:
                st.error("❌ Model not loaded. Please check model file.")
            else:
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
            if model is None:
                st.error("❌ Model not loaded. Please check model file.")
            else:
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
