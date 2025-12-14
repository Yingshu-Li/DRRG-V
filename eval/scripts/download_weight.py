import os
from huggingface_hub import snapshot_download

# ================= 配置区域 =================
# 1. 模型 ID
REPO_ID = "google/siglip2-so400m-patch14-384"

# 2. 你想要保存的指定服务器路径 (请修改这里)
# 例如: "/data/models/Qwen3" 或 "D:\\Models\\Qwen3"
LOCAL_DIR = "/mnt/sdb/pretrained_models/siglip2-so400m-patch14-384"

# 3. Token (如果是私有模型需要填写，公开模型留空即可)
HF_TOKEN = None 
# ===========================================

def download_repo():
    print(f"🚀 开始下载模型: {REPO_ID}")
    print(f"📂 目标路径: {LOCAL_DIR}")

    try:
        # 创建目录（如果不存在）
        os.makedirs(LOCAL_DIR, exist_ok=True)

        # 开始下载
        # local_dir_use_symlinks=False 确保下载的是实际文件而不是缓存的快捷方式
        path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False, 
            token=HF_TOKEN,
            resume_download=True  # 支持断点续传
        )
        print(f"✅ 下载完成！模型已保存在: {path}")
        
    except Exception as e:
        print(f"❌ 下载出错: {e}")

if __name__ == "__main__":
    download_repo()