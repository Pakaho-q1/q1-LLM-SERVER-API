#!/usr/bin/env python3
"""
upload_to_github.py - อัปโหลดโปรเจกต์ขึ้น GitHub อัตโนมัติ
วิธีใช้: python upload_to_github.py [project_path]

config file: อยู่ที่เดียวกับสคริปต์นี้ ชื่อ git-upload-config.json
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import fnmatch


def get_script_dir():
    """หาพาธของโฟลเดอร์ที่เก็บสคริปต์นี้"""
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    """โหลด config จากโฟลเดอร์เดียวกับสคริปต์"""
    script_dir = get_script_dir()
    config_path = os.path.join(script_dir, "git-upload-config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # ถ้าไม่มีไฟล์ config ให้ใช้ค่าเริ่มต้น
        print(f"⚠️ ไม่พบไฟล์ config ที่ {config_path}")
        print("📝 กำลังใช้ค่าเริ่มต้น...")
        return {
            "ignore_patterns": ["__pycache__/", "*.log", "temp/", ".env", ".git/"],
            "default_branch": "main",
            "remote_name": "origin",
            "auto_commit_message": "Auto commit on {date}",
            "ask_for_message": True,
        }


def is_ignored(rel_path, patterns):
    """ตรวจสอบว่าไฟล์/โฟลเดอร์ (พาธสัมพัทธ์) ตรงกับ ignore pattern หรือไม่"""
    rel_path = rel_path.replace(os.sep, "/")
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or rel_path.startswith(
            pattern.rstrip("/") + "/"
        ):
            return True
    return False


def get_files_to_add(base_path, ignore_patterns):
    """หาไฟล์ทั้งหมดใน base_path ที่ไม่ถูก ignore"""
    to_add = []
    for root, dirs, files in os.walk(base_path):
        # สร้างพาธสัมพัทธ์จาก base_path
        rel_root = os.path.relpath(root, base_path)
        if rel_root == ".":
            rel_root = ""
        else:
            rel_root = rel_root.replace(os.sep, "/")

        # ข้าม .git folder (ป้องกันไว้ แม้จะใส่ใน ignore_patterns แล้ว)
        if ".git" in dirs:
            dirs.remove(".git")

        # กรอง directories ตาม ignore_patterns
        dirs_to_remove = []
        for d in dirs:
            rel_d = os.path.join(rel_root, d) if rel_root else d
            if is_ignored(rel_d, ignore_patterns):
                dirs_to_remove.append(d)
        for d in dirs_to_remove:
            dirs.remove(d)

        # เพิ่มไฟล์ที่ไม่ถูก ignore
        for f in files:
            rel_f = os.path.join(rel_root, f) if rel_root else f
            if not is_ignored(rel_f, ignore_patterns):
                to_add.append(os.path.join(root, f))
    return to_add


def run_git_command(cmd, cwd):
    """รันคำสั่ง git ในโฟลเดอร์ที่กำหนด คืนค่า output ถ้าสำเร็จ"""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Git error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def print_banner():
    """แสดงแบนเนอร์สวยงาม"""
    print("=" * 60)
    print("  🚀 GitHub Auto Uploader")
    print("=" * 60)


def main():
    print_banner()

    # รับพาธโปรเจกต์จาก argument
    if len(sys.argv) > 1:
        project_path = os.path.abspath(sys.argv[1])
    else:
        project_path = os.getcwd()
        print(f"📌 ไม่ได้ระบุพาธ ใช้โฟลเดอร์ปัจจุบัน")

    if not os.path.isdir(project_path):
        print(f"❌ Error: '{project_path}' ไม่ใช่โฟลเดอร์ที่ถูกต้อง")
        sys.exit(1)

    print(f"📂 โปรเจกต์: {project_path}")
    print(f"⚙️  Config: {os.path.join(get_script_dir(), 'git-upload-config.json')}")

    # โหลด config (จากโฟลเดอร์เดียวกับสคริปต์)
    config = load_config()
    ignore_patterns = config.get("ignore_patterns", [])
    remote_name = config.get("remote_name", "origin")
    default_branch = config.get("default_branch", "main")
    auto_msg_template = config.get("auto_commit_message", "Auto commit on {date}")
    ask_for_msg = config.get("ask_for_message", True)

    # แสดง ignore patterns ที่กำลังใช้
    if ignore_patterns:
        print("🔇 Ignore patterns:")
        for p in ignore_patterns:
            print(f"   - {p}")

    # 1. ตรวจสอบ Git
    print("\n🔍 ตรวจสอบ Git...")
    try:
        git_version = run_git_command(["git", "--version"], cwd=project_path)
        print(f"   ✅ {git_version}")
    except:
        print("   ❌ กรุณาติดตั้ง Git ก่อน")
        sys.exit(1)

    # 2. ตรวจสอบว่าเป็น Git repo หรือไม่
    git_dir = os.path.join(project_path, ".git")
    if not os.path.exists(git_dir):
        print("🔄 กำลังสร้าง Git repository...")
        run_git_command(["git", "init"], cwd=project_path)
        print("   ✅ สร้าง repository แล้ว")
    else:
        print("   ✅ มี Git repository อยู่แล้ว")

    # 3. ตรวจสอบ remote origin
    print("\n🌐 ตรวจสอบ remote...")
    try:
        remote_url = run_git_command(
            ["git", "remote", "get-url", remote_name], cwd=project_path
        )
        print(f"   ✅ พบ remote '{remote_name}': {remote_url}")
    except:
        print(f"   ⚠️ ไม่พบ remote '{remote_name}'")
        repo_url = input("   กรุณาใส่ URL ของ repository (หรือกด Enter เพื่อข้าม): ").strip()
        if repo_url:
            run_git_command(
                ["git", "remote", "add", remote_name, repo_url], cwd=project_path
            )
            print(f"   ✅ เพิ่ม remote '{remote_name}' แล้ว")
        else:
            print("   ℹ️ ไม่มี remote ดำเนินการเฉพาะ local commit เท่านั้น")

    # 4. เพิ่มไฟล์ที่ต้องการ
    print("\n📁 กำลังตรวจสอบไฟล์...")
    files_to_add = get_files_to_add(project_path, ignore_patterns)
    if files_to_add:
        # แสดงตัวอย่างไฟล์ที่จะเพิ่ม (สูงสุด 5 ไฟล์)
        print(f"   พบไฟล์ที่จะเพิ่ม {len(files_to_add)} ไฟล์:")
        for i, f in enumerate(files_to_add[:5]):
            rel_f = os.path.relpath(f, project_path)
            print(f"     - {rel_f}")
        if len(files_to_add) > 5:
            print(f"     ... และอีก {len(files_to_add) - 5} ไฟล์")

        # เพิ่มไฟล์ทีละไฟล์ (เพื่อให้ git add ทำงานถูกต้อง)
        for f in files_to_add:
            rel_f = os.path.relpath(f, project_path)
            run_git_command(["git", "add", "--", rel_f], cwd=project_path)
        print(f"   ✅ เพิ่มไฟล์เรียบร้อย")
    else:
        print("   ℹ️ ไม่มีไฟล์ที่ต้องเพิ่ม (อาจถูก ignore หมด)")

    # 5. Commit ถ้ามีการเปลี่ยนแปลง
    print("\n💾 ตรวจสอบการเปลี่ยนแปลง...")
    status = run_git_command(["git", "status", "--porcelain"], cwd=project_path)
    if not status:
        print("   ℹ️ ไม่มีการเปลี่ยนแปลง ไม่ต้อง commit")
    else:
        # แสดงสถานะ
        changed_files = status.count("\n") + 1
        print(f"   มีไฟล์เปลี่ยนแปลง {changed_files} ไฟล์")

        if ask_for_msg:
            msg = input("   💬 ข้อความ commit (หรือ Enter เพื่อใช้อัตโนมัติ): ").strip()
        else:
            msg = ""
        if not msg:
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = auto_msg_template.format(date=date_str)

        run_git_command(["git", "commit", "-m", msg], cwd=project_path)
        print(f'   ✅ Commit: "{msg}"')

    # 6. Push (ถ้ามี remote)
    print("\n📤 กำลัง push...")
    try:
        remote_url = run_git_command(
            ["git", "remote", "get-url", remote_name], cwd=project_path
        )
        # หา branch ปัจจุบัน
        current_branch = run_git_command(
            ["git", "branch", "--show-current"], cwd=project_path
        )
        branch = current_branch if current_branch else default_branch
        print(f"   🚀 กำลัง push ไปยัง {remote_name}/{branch}...")
        run_git_command(["git", "push", "-u", remote_name, branch], cwd=project_path)
        print(f"   ✅ Push สำเร็จ!")
    except:
        print("   ℹ️ ข้ามขั้นตอน push (ไม่มี remote หรือเกิดข้อผิดพลาด)")

    print("\n" + "=" * 60)
    print("  ✨ เสร็จสิ้น! ✨")
    print("=" * 60)


if __name__ == "__main__":
    main()
