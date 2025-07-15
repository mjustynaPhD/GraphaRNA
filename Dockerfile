FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY . .
COPY RiNALMo /app/RiNALMo

RUN pip install --upgrade pip setuptools wheel build

RUN pip install . --no-deps

RUN pip install torch==2.3.0 \
    torch-geometric==2.5.3 \
    numpy==1.26.4 \
    scikit-learn>=1.4.0 \
    pandas \
    biopython>=1.83 \
    rnapolis==0.3.11 \
    wandb \
    torch-scatter==2.1.2+pt23cu121 \
    torch-sparse==0.6.18+pt23cu121 \
    torch-cluster==1.6.3+pt23cu121 \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html \
    einops

RUN pip install ./RiNALMo

CMD ["grapharna", "--input=user_inputs/tsh_helix.dotseq"]
