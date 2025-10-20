from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastai.vision.all import *
from huggingface_hub import hf_hub_download
import pandas as pd
import io
import os
from PIL import Image

app = FastAPI(title="IntellisScan API", version="1.0")

# Download model if not exists
def load_model():
    if not os.path.exists("intelliscan_model_final.pkl"):
        print("📥 Downloadimg model from Hugging Face Hub...")
        model_path = hf_hub_download(
            repo_id="ashvinkumar/intelliscan-model",
            filename="intelliscan_model_final.pkl",
            local_dir="."
        )
        return model_path
    return "intelliscan_model_final.pkl"

model_path = load_model()
model = load_learner(model_path)

@app.post("/predict-single")
async def predict_single(file: UploadFile = File(...)):
    try:
        #read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # save temp file 
        temp_path = f"temp_{file.filename}"
        image.save(temp_path)

        #predict
        prediction, _, probs = model.predict(temp_path)
        confidence = probs.max().item()
        
        #clean up
        os.remove(temp_path)

        return {
            "document_type": str(prediction),
            "confidence": confidence,
            "all_probabilities": {model.dls.vocab[i]: float(probs[i]) for i in range(len(model.dls.vocab))},
            "status":"success"
        }
    except Exception as e:
        return {"error": str(e), "status":"error"}

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            # Read and process each file directly
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            temp_path = f"temp_{file.filename}"
            image.save(temp_path)
            
            prediction, _, probs = model.predict(temp_path)
            confidence = probs.max().item()
            os.remove(temp_path)
            
            results.append({
                "filename": file.filename,
                "document_type": str(prediction),
                "confidence": confidence
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "document_type": "Error",
                "confidence": 0.0,
                "error": str(e)
            })
    return {"results": results, "total_processed": len(results)}


@app.get("/export-csv") # export functionailty
async def export_csv(results: list):
    df = pd.DataFrame(results)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )




@app.get("/")
async def root():
    return {"message": "IntelliScan API - Document Classification"}