import os

LOG_FILE = "/var/log/syslog"
OUTPUT_FILE = "error_report.txt"

def analyze_logs():
    # 1. 检查日志文件是否存在
    if not os.path.exists(LOG_FILE):
        print(f"错误: 找不到日志文件 {LOG_FILE}")
        return

    # 2. 读取上一次报告的“最后一条错误”，作为本次扫描的停止标记
    last_known_error = None
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f_old:
                lines = f_old.readlines()
                # 获取非空行中的最后一行（跳过类似 "共发现 X 条错误记录:" 的标题行）
                for line in reversed(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("共发现"):
                        last_known_error = stripped
                        break
        except Exception as e:
            print(f"读取旧报告失败: {e}")

    error_lines = []
    
    # 3. 从文件末尾开始读取，只获取增量错误
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            # 将文件指针移动到文件末尾
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            
            # 从末尾向前逐字节读取
            while file_size > 0:
                # 每次向前移动一个字节
                file_size -= 1
                f.seek(file_size)
                char = f.read(1)
                
                # 如果读到换行符，说明找到了一整行
                if char == '\n':
                    # 读取当前指针到行尾的内容
                    line = f.readline().strip()
                    
                    # 核心逻辑：如果当前行等于上次记录的最后一条错误，说明已经读取完毕，直接跳出
                    if line == last_known_error:
                        break
                    
                    # 否则，判断是否包含错误关键字
                    if "error" in line.lower() or "fail" in line.lower():
                        error_lines.append(line)
                        
    except PermissionError:
        print("权限不足! 请使用管理员权限运行此脚本 (sudo python3 script.py)。")
        return

    # 4. 将新发现的错误行写入文件（倒序还原为正常时间顺序）
    if error_lines:
        error_lines.reverse()
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            f_out.write(f"本次新增 {len(error_lines)} 条错误记录:\n")
            f_out.writelines([line + "\n" for line in error_lines])
        print(f"分析完成！已将 {len(error_lines)} 条新增错误记录保存至 {OUTPUT_FILE}")
    else:
        print("太棒了！自上次运行以来，没有发现新的错误记录。")

if __name__ == "__main__":
    analyze_logs()