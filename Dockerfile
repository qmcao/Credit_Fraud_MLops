FROM python:3.10-slim

WORKDIR /application

COPY . /application

# Install system dependency for OpenMP (needed by LightGBM)
RUN apt update -y \
 && apt install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*


# Install AWS CLI via pip (no apt needed)
RUN pip install --no-cache-dir awscli

# Install your other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "application.py"]
