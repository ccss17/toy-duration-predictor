# Step 1: Specify the base OS and Python version.
FROM python:3.12-slim

# Step 2: Set the working directory inside the container.
WORKDIR /app

# Step 3: Install git, which is required by pip to install from a git repo.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Step 4: Copy your requirements.txt file into the container.
COPY requirements.txt .

# Step 5: Install all the Python packages listed in your requirements.txt.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Step 6: Copy the rest of your project files (like your notebook and src folder).
COPY . .

# Step 7: Define the default command to run when the Space starts.
# --- THE FIX IS HERE ---
# We add '--allow-origin=*' to relax the Cross-Origin Resource Sharing (CORS) policy
# and '--ServerApp.allow_remote_access=True' to ensure the server is accessible.
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=7860", "--allow-root", "--NotebookApp.token=''", "--ServerApp.allow_origin='*'", "--ServerApp.allow_remote_access=True"]