import os
import shutil

# 文件夹路径
labels_dir = 'C:\\Users\\gaor\\Desktop\\vision\\datasets\\cfimg\\labels'
images_dir = 'C:\\Users\\gaor\\Desktop\\vision\\datasets\\cfimg\\all_images'
select_dir = 'C:\\Users\\gaor\\Desktop\\vision\\datasets\\cfimg\\select'

# 如果 select 文件夹不存在，创建它
if not os.path.exists(select_dir):
    os.makedirs(select_dir)

# 获取labels目录中的所有txt文件
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
i=0
# 遍历所有label文件
for label_file in label_files:
    # 获取文件名（不带扩展名）
    file_base_name = os.path.splitext(label_file)[0]
    # 构造对应的jpg文件名
    image_file = file_base_name + '.jpg'
    # 构造图片文件的完整路径
    image_path = os.path.join(images_dir, image_file)
    # 检查对应的jpg文件是否存在
    if os.path.exists(image_path):
        # 将jpg文件复制到select目录
        shutil.copy(image_path, select_dir)
        i+=1
        print(i)

print("匹配的文件已复制到 select 文件夹中。")
