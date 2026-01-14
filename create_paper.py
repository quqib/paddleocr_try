import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import os

# 步骤1：加载原始图片
image_path = "2.jpg"  # 替换为你的图片路径
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("无法加载图片，请检查路径")

# 步骤2：预处理 —— 提取纸张背景颜色
# 1. 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 使用自适应阈值获取文字区域（黑色文字）
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# 3. 找到文字区域轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # 获取最大轮廓（通常是整页文字）
    largest_contour = max(contours, key=cv2.contourArea)
    # 扩展一点，覆盖整个纸张
    x, y, w, h = cv2.boundingRect(largest_contour)
    padding = 20
    x, y, w, h = max(0, x-padding), max(0, y-padding), min(w+2*padding, img.shape[1]), min(h+2*padding, img.shape[0])
    paper_roi = img[y:y+h, x:x+w]
else:
    paper_roi = img  # 备用方案

# 4. 提取背景颜色（去除文字区域，取平均）
# 方法：使用高斯模糊 + 色彩统计
blurred = cv2.GaussianBlur(paper_roi, (15, 15), 0)
avg_color_bgr = np.mean(blurred, axis=(0, 1)).astype(int)
avg_color_rgb = avg_color_bgr[::-1]  # BGR -> RGB
print(f"纸张平均颜色 (RGB): {avg_color_rgb}")

# 步骤3：创建 A4 纸尺寸图像（72 DPI）
a4_width = int(8.27 * 72)  # 595 px
a4_height = int(11.69 * 72)  # 842 px

# 创建 PIL 图像
new_img = Image.new('RGB', (a4_width, a4_height), color=tuple(avg_color_rgb))

# 步骤4：添加轻微渐变（模拟光照效果）
draw = ImageDraw.Draw(new_img)

# 创建渐变蒙版（顶部亮，底部稍暗）
gradient = Image.new('L', (a4_width, a4_height))
for y in range(a4_height):
    brightness = int(255 * (1 - y / a4_height) * 0.15)  # 15% 的亮度变化
    for x in range(a4_width):
        gradient.putpixel((x, y), 255 - brightness)

# 应用渐变叠加（柔光混合）
# new_img = Image.blend(new_img, new_img.convert('RGBA'), 0.95)

# 或者更自然地：用渐变叠加一层
overlay = Image.new('RGBA', (a4_width, a4_height), (255, 255, 255, 50))  # 半透明白
grad_overlay = Image.new('L', (a4_width, a4_height))
for y in range(a4_height):
    grad_val = int(255 * (1 - y / a4_height) * 0.2)
    grad_overlay.putpixel((0, y), grad_val)
grad_overlay = grad_overlay.resize((a4_width, a4_height))
grad_overlay = grad_overlay.convert('RGBA')
grad_overlay.putalpha(50)
new_img.paste(grad_overlay, (0, 0), grad_overlay)

# 步骤5：保存结果
new_img.save("generated_a4_paper_similar_background.png")
print("✅ 已生成相似背景的A4纸张图像！")