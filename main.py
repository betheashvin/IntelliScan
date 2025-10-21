from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastai.vision.all import *
import pandas as pd
import io
import os
import requests
from PIL import Image

app = FastAPI(title="IntelliScan API", version="1.0")

# Download model if not exists
def load_model():

    model_path = "intelliscan_model_final.pkl"

    if not os.path.exists(model_path):
        print("📥 Downloadimg model from Google Drive...")

        file_id = "1HGduInluShD47GUvVYUeDXKqHyjhP0wv"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            reponse = requests.get(download_url)
            with open(model_path, "wb") as f:
                f.write(reponse.content)
            print(" ✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return None
        
    return model_path

model_path = load_model()
if model_path:
    model = load_learner(model_path)
else:
    raise Exception("Model could not be loaded.")

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
