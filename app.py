from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, PlainTextResponse
import uuid
import os
import subprocess
from time import sleep

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OK"}

@app.post("/run")
async def run_grapharna(uuid: str = Form(...), seed: int = Form(42)):
    input_path = f"/shared/user_inputs/{uuid}.dotseq"
    output_path = f"/shared/samples/grapharna-seed={seed}/800/{uuid}.pdb"

    try:
        subprocess.run([
            "grapharna",
            f"--input={input_path}",
            f"--seed={seed}"
        ], check=True)

        return FileResponse(output_path, media_type="text/plain", filename=f"{uuid}.pdb")

    except subprocess.CalledProcessError as e:
        return {"error": f"Grapharna failed: {e}"}
    

@app.post("/test")
async def test_stub():
    seed = 42
    input_path = "/shared/user_inputs/test.dotseq"
    with open(input_path, "r") as f:
        tekst = f.read()

    output_dir = f"/shared/samples/grapharna-seed={seed}/800/"
    output_path = f"/shared/samples/grapharna-seed={seed}/800/test.pdb"

    os.makedirs(output_dir, exist_ok=True)

    # Symulacja obliczeń
    sleep(5)

    with open(output_path, "w") as f:
        f.write(tekst)

    return PlainTextResponse(output_path)