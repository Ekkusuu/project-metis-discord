## Local Llama.cpp Integration

The backend uses `llama-cpp-python` to run the local GGUF model located at `Model/dolphin-2.6-mistral-7b.Q5_K_M.gguf`.

### Backend Setup

1. Create & activate a virtual environment (optional but recommended).
2. Install requirements:
	 ```cmd
	 pip install -r requirements.txt
	 ```
3. Run the backend:
	 ```cmd
	 python backend\main.py
	 ```

### Windows Notes (llama-cpp-python)
`llama-cpp-python` will attempt to build native extensions. If you encounter build errors:

* Ensure you have CMake installed and in PATH.
* Install Visual Studio Build Tools (C++ workload) or full VS 2022.
* Optional performance flags (set before `pip install`):
	```cmd
	set CMAKE_ARGS=-DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_FAST=ON
	set FORCE_CMAKE=1
	pip install llama-cpp-python --force-reinstall --no-cache-dir
	```

If you prefer a prebuilt wheel, you can try:
```cmd
pip install llama-cpp-python --prefer-binary
```

### Frontend Setup
Install dependencies and run Vite dev server:
```cmd
cd frontend
npm install
npm run dev
```

If backend runs on a different host/port, create `frontend/.env`:
```
VITE_API_URL=http://localhost:8000
```

### Chat API
POST `http://localhost:8000/chat`
Body example:
```json
{
	"messages": [
		{"role": "system", "content": "You are Metis, a helpful AI assistant."},
		{"role": "user", "content": "Hello!"}
	],
	"max_tokens": 256
}
```
Response:
```json
{"reply": "Hi there! How can I help you today?"}
```

### Environment Tweaks
You can tune context and threads:
```cmd
set LLAMA_CTX=4096
set LLAMA_THREADS=8
```
Before launching the backend.
# project-metis