import subprocess
import sys
from pathlib import Path

MODEL_PATH = "./models/qwen3-14b-qlora-soulchat"   # 改成你的模型目录
SERVED_NAME = "qwen3-14b-soulchat"
HOST = "0.0.0.0"
PORT = "6006"
API_KEY = "sk-local-change-me"  # 你也可以不设，但建议设
TEMPLATE = "./qwen3_nonthinking.jinja"

assert Path(MODEL_PATH).exists(), f"MODEL_PATH not found: {MODEL_PATH}"

cmd = [
    "vllm", "serve", MODEL_PATH,
    "--served-model-name", SERVED_NAME,
    "--host", HOST,
    "--port", PORT,
    "--dtype", "auto",
    "--api-key", API_KEY,
    "--chat-template", TEMPLATE,
    "--gpu-memory-utilization", "0.97",
    "--max-num-seqs", "1",
    "--max-model-len", "8192",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    #"--max-num-batched-tokens", "512",
    #"--cpu-offload-gb", "8",
]

print("Launching:", " ".join(cmd))
# 阻塞运行；如果你想后台运行可以用 subprocess.Popen
subprocess.run(cmd, check=True)
