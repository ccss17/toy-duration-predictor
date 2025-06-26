# Step 1: Specify the base OS and Python version.
FROM python:3.12-slim

# --- THE FIX IS HERE: Install system dependencies as root FIRST ---

# Step 2: Update package lists and install the `git` command-line tool.
# This must be done as the root user before switching to the unprivileged user.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Step 3: Now, create the non-root user and its home directory.
RUN useradd -m -u 1000 user

# Step 4: Switch to the new user.
USER user

# Step 5: Set environment variables for the new user.
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# Step 6: Copy the requirements file and install packages as the new user.
# This will now succeed because git is installed.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Step 7: Copy the rest of your project files.
COPY --chown=user:user . .

# Step 8: Define the default command to run when the Space starts.
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=7860", "--NotebookApp.token=''"]