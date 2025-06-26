FROM python:3.12-slim
# Set the working directory
# All subsequent commands will run from here.
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
# Copy project files into the container.
# This copies everything from your local directory into the container's /app directory.
COPY . .
# Install all the Python packages listed in your requirements.txt.
# The --no-cache-dir flag keeps the final image size smaller.
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# Define the default command to run when the Space starts.
# This launches a JupyterLab server that is accessible from the web.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=7860", "--allow-root", "--NotebookApp.token=''"]