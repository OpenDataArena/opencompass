import os
import shutil
from pathlib import Path

# --- 配置 ---
INPUT_ROOT_DIR_NAME = "outputs_rea"  # 输入的根文件夹
OUTPUT_DIR_NAME = "res"        # 输出文件夹

def process_and_copy_files():
    """
    遍历 outputs/ 文件夹，找到符合条件的 CSV 文件，并将其复制到 res/ 文件夹下。
    """
    input_root_dir = Path(INPUT_ROOT_DIR_NAME)
    output_dir = Path(OUTPUT_DIR_NAME)

    if not input_root_dir.is_dir():
        print(f"错误：输入文件夹 '{input_root_dir}' 不存在或不是一个目录。")
        return

    # 确保输出文件夹存在，如果不存在则创建
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出文件夹 '{output_dir}' 已准备就绪。")

    copied_files_count = 0

    # 1. 遍历 outputs/ 下的第一层文件夹 (例如: sft-autocot-seed-4442_20250519-11:23:25_llama_test)
    for first_level_dir in input_root_dir.iterdir():
        if not first_level_dir.is_dir():
            continue

        # 提取前缀 (例如: sft-autocot-seed-4442)
        # 假设前缀是第一个下划线之前的部分
        parts = first_level_dir.name.split('_', 1)
        if not parts: # 文件夹名为空或不含下划线，不太可能但做个检查
            continue
        prefix = parts[0]

        # 2. 遍历第一层文件夹下的第二层文件夹 (唯一ID文件夹, 例如: 20250519_125642)
        for unique_id_dir in first_level_dir.iterdir():
            if not unique_id_dir.is_dir():
                continue

            unique_id = unique_id_dir.name # 这就是唯一ID

            # 3. 构建期望的summary文件夹和CSV文件路径
            # 路径: outputs/.../unique_id/summary/summary_unique_id.csv
            expected_summary_dir = unique_id_dir / "summary"
            expected_csv_filename = f"summary_{unique_id}.csv"
            source_csv_path = expected_summary_dir / expected_csv_filename

            if source_csv_path.is_file():
                # 4. 构建目标文件名和路径
                # 目标文件名: 前缀-唯一id.csv (例如: sft-autocot-seed-4442-20250519_125642.csv)
                destination_filename = f"{prefix}-{unique_id}.csv"
                destination_path = output_dir / destination_filename

                # 5. 拷贝文件
                try:
                    shutil.copy2(source_csv_path, destination_path) # copy2 会保留元数据
                    print(f"已拷贝: '{source_csv_path}' -> '{destination_path}'")
                    copied_files_count += 1
                except Exception as e:
                    print(f"拷贝文件 '{source_csv_path}' 时出错: {e}")
            # else:
                # print(f"调试: 未找到或不是文件: {source_csv_path}") # 可用于调试

    if copied_files_count > 0:
        print(f"\n成功拷贝 {copied_files_count} 个文件到 '{output_dir}' 文件夹。")
    else:
        print(f"\n未在 '{input_root_dir}' 中找到符合条件的CSV文件进行拷贝。")

def create_dummy_structure_for_testing():
    """
    创建一个用于测试的伪文件结构 (可选)
    """
    print("正在创建用于测试的伪文件结构...")
    base_outputs = Path(INPUT_ROOT_DIR_NAME)
    base_outputs.mkdir(exist_ok=True)

    # 第一个合法结构
    path1_summary = base_outputs / "sft-autocot-seed-4442_20250519-11:23:25_llama_test" / "20250519_125642" / "summary"
    path1_summary.mkdir(parents=True, exist_ok=True)
    (path1_summary / "summary_20250519_125642.csv").write_text("header1,header2\ndata1,data2")
    print(f"已创建: {path1_summary / 'summary_20250519_125642.csv'}")

    # 第二个合法结构 (不同前缀和ID)
    path2_summary = base_outputs / "my-other-prefix_timestamp_model" / "20240101_000000" / "summary"
    path2_summary.mkdir(parents=True, exist_ok=True)
    (path2_summary / "summary_20240101_000000.csv").write_text("colA,colB\nvalA,valB")
    print(f"已创建: {path2_summary / 'summary_20240101_000000.csv'}")

    # 一个不包含 summary 文件夹的结构
    (base_outputs / "no-summary-folder_ts" / "id123").mkdir(parents=True, exist_ok=True)
    print(f"已创建 (无summary): {base_outputs / 'no-summary-folder_ts' / 'id123'}")

    # 一个 summary 文件夹中没有对应CSV的结构
    path3_summary_no_csv = base_outputs / "prefix3_ts" / "id789" / "summary"
    path3_summary_no_csv.mkdir(parents=True, exist_ok=True)
    (path3_summary_no_csv / "other_file.txt").write_text("some text")
    print(f"已创建 (summary但无对应csv): {path3_summary_no_csv}")

    print("伪文件结构创建完毕。")

if __name__ == "__main__":
    # 如果你需要测试，可以取消下面这行注释来创建伪文件结构
    # create_dummy_structure_for_testing()

    # 执行主要的文件处理和拷贝逻辑
    process_and_copy_files()

    # (可选) 如果你创建了伪文件，并且想在测试后删除它们：
    # if Path(INPUT_ROOT_DIR_NAME).exists() and "dummy" in INPUT_ROOT_DIR_NAME.lower(): # 安全检查
    #     shutil.rmtree(INPUT_ROOT_DIR_NAME)
    #     print(f"已删除测试文件夹 '{INPUT_ROOT_DIR_NAME}'")
    # if Path(OUTPUT_DIR_NAME).exists():
    #     # 你可能想检查 res 文件夹是否只包含本次运行生成的文件再删除
    #     # shutil.rmtree(OUTPUT_DIR_NAME)
    #     # print(f"已删除输出文件夹 '{OUTPUT_DIR_NAME}'")
    #     pass

