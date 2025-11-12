import fnmatch
import os


#def rename_matched_folders(directory, old_str, new_str):
#    """
#    递归重命名与old_str完全相同的文件夹名称
#    返回是否进行了任何重命名操作
#    """
#    renamed = False
#    for root, dirs, files in os.walk(directory, topdown=False):
#        for dir_name in dirs:
#            if dir_name == old_str:
#                old_path = os.path.join(root, dir_name)
#                new_path = os.path.join(root, new_str)
#
#                try:
#                    os.rename(old_path, new_path)
#                    print(f"[文件夹重命名] {os.path.relpath(old_path, directory)} -> {new_str}")
#                    renamed = True
#                except Exception as e:
#                    print(f"[错误] 无法重命名文件夹 {old_path}: {str(e)}")
#    return renamed
def rename_matched_items(directory, old_str, new_str):
    """
    递归重命名所有包含 old_str 的文件夹和文件名称，将其中的 old_str 替换为 new_str
    返回是否进行了任何重命名操作
    """
    renamed = False

    # 先处理文件
    for root, dirs, files in os.walk(directory, topdown=False):
        for file_name in files:
            if old_str in file_name:
                new_name = file_name.replace(old_str, new_str)
                old_path = os.path.join(root, file_name)
                new_path = os.path.join(root, new_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"[文件重命名] {os.path.relpath(old_path, directory)} -> {new_name}")
                    renamed = True
                except Exception as e:
                    print(f"[错误] 无法重命名文件 {old_path}: {str(e)}")

    # 再处理文件夹（topdown=False 确保先处理子目录）
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            if old_str in dir_name:
                new_name = dir_name.replace(old_str, new_str)
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"[文件夹重命名] {os.path.relpath(old_path, directory)} -> {new_name}")
                    renamed = True
                except Exception as e:
                    print(f"[错误] 无法重命名文件夹 {old_path}: {str(e)}")

    return renamed


def replace_string_in_files(directory, old_str, new_str, file_pattern="*"):
    """
    递归替换目录及其子目录中所有文件的内容
    并重命名匹配的文件夹名称
    """
    # 将相对路径转换为绝对路径
    abs_directory = os.path.abspath(directory)

    if not os.path.exists(abs_directory):
        print(f"错误：目录不存在 '{abs_directory}'")
        return

    print(f"正在扫描目录: {abs_directory}")

    # 先重命名文件夹
    folder_renamed = rename_matched_items(abs_directory, old_str, new_str)

    # 然后处理文件内容
    for root, dirs, files in os.walk(abs_directory):
        for filename in fnmatch.filter(files, file_pattern):
            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, abs_directory)

            try:
                # 读取文件内容
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()

                # 替换字符串
                new_content = content.replace(old_str, new_str)

                # 如果内容有变化，则写入文件
                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    print(f"[文件已修改] {relative_path}")
                else:
                    print(f"[文件无更改] {relative_path}")

            except UnicodeDecodeError:
                # 处理非文本文件或不同编码的文件
                print(f"[跳过] {relative_path} (非文本文件或编码不匹配)")
            except PermissionError:
                print(f"[错误] {relative_path} (权限不足)")
            except Exception as e:
                print(f"[错误] {relative_path} ({str(e)})")

    if not folder_renamed:
        print("[提示] 没有找到需要重命名的文件夹")


if __name__ == "__main__":
    directory = "."
    new_str = "data_analysis_with_agent"
    old_str = "data_analysis_with_agent"
    file_pattern = "*"

    print("=== 文件和文件夹批量替换工具 ===")
    print("功能:")
    print("1. 替换文件内容中的指定字符串")
    print("2. 重命名与指定字符串完全相同的文件夹")

    if not directory:
        print("错误：必须指定目录路径")
        exit(1)

    if not old_str:
        print("错误：必须指定要替换的字符串")
        exit(1)

    print("\n即将执行替换操作:")
    print(f"目录: {os.path.abspath(directory)}")
    print(f"将 '{old_str}' 替换为 '{new_str}'")
    print(f"文件类型: {file_pattern if file_pattern != '*' else '所有文件'}")
    print("注意：与字符串完全相同的文件夹也将被重命名")

    confirm = input("\n确定要执行替换吗？(y/n): ").lower()
    if confirm != "y":
        print("操作已取消")
        exit(0)

    print("\n开始处理...\n")
    replace_string_in_files(directory, old_str, new_str, file_pattern)
    print("\n操作完成!")
