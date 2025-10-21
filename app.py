import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import base64

st.set_page_config(page_title="IntelliScan", layout="centered")
st.title("IntelliScan - AI Document Classifier")
st.write("Upload documents to classify as Invoice, Receipt, Contract, or Research Paper ")

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
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post("http://localhost:8000/predict-single",files=files)

                    if response.status_code == 200:
                        result = response.json()

                        st.success(" Classififcation Complete")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Document Type", result['document_type'].title())
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2%}")

                        st.subheader("Detailed Analysis:")
                        for doc_type, prob in result['all_probabilities'].items():
                            st.progress(prob, text=f'{doc_type.title()}: {prob:.2%}')   

                    else:
                        st.error("Classfication Failed") 

                except Exception as e:
                    st.error(f"Connection error: {e}")

with tab2:
    # Batch processing
    st.subheader("Process Multiple Files")
    uploaded_files = st.file_uploader("Choose multiple files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")

    if uploaded_files:
        st.write(f"📁 Selected {len(uploaded_files)} files")

        if st.button("🔍 Classify All Documents", type="primary", key="batch_btn"):
            results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                try:
                    #Prcoess each file
                    files = {"file": file.getvalue()}
                    response = requests.post("http://localhost:8000/predict", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            "filename": file.name,
                            "document_type": result['document_type'],
                            "confidence": f"{result['confidence']:.2%}",
                            "status":"✅ Success"
                        })
                    else:
                        results.append({
                            "filename": file.name,
                            "document_type":"Failed",
                            "confidence":"N/A",
                            "status": "❌ Error"
                        })

                    progress_bar.progress((i + 1) / len(uploaded_files))

                except Exception as e:
                    results.append({
                        "filename":file.name,
                        "document_type":"Failed",
                        "Confidence": "0%",
                        "status": f"❌ Error: {e}"    
                    })

            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 Results Summary")
                st.dataframe(df)

                #Export To CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="intelliscan_results.csv",
                    mime="text/csv"
                )

#Footer
st.markdown("---")
st.markdown("**Model Accuracy:** 95%+ | **Supported Types:** Invoices, Receipts, Contracts, Research Papers | **Features:** Single & Batch Processing")

