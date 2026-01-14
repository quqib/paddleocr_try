import cv2
import numpy as np

img = cv2.imread("2.jpg")

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 这里简单用阈值，黑色区域当作缺失部分（你可以自己调）
_, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

# 去噪（让mask更干净）
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 修复
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite("repair.jpg", dst)
cv2.imwrite("mask.jpg", mask)  # 保存 mask 用于检查

# 下面这些在无 GUI 的 OpenCV 中不能用
# cv2.imshow("mask", mask)
# cv2.imshow("result", dst)
# cv2.waitKey(0)



exit()
import cv2
import numpy as np

# 读取图像
img = cv2.imread('2.jpg')  # 替换为你自己的文件名
original = img.copy()

# 1. 转灰度 + 高斯模糊 + 边缘检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 2. 寻找轮廓，找到最大轮廓（即纸张）
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # 取最大轮廓

if len(contours) == 0:
    print("未找到轮廓")
else:
    contour = contours[0]

    # 3. 拟合四边形（透视变换）
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)

    # 4. 排序四个点（左上、右上、右下、左下）
    def order_points(pts):
        pts = pts.astype(np.float32)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        return rect

    ordered = order_points(box)

    # 5. 获取透视变换矩阵
    (tl, tr, br, bl) = ordered
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 6. 调整到标准 A4 尺寸（可选）
    # 标准 A4 分辨率 300 dpi ≈ 2480×3508 像素
    target_width = 2480
    target_height = 3508

    # 缩放至标准尺寸（保持比例）
    scale = min(target_width / warped.shape[1], target_height / warped.shape[0])
    new_w = int(warped.shape[1] * scale)
    new_h = int(warped.shape[0] * scale)

    resized = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 7. 创建标准 A4 图像（纯白背景）
    a4_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255  # 白色背景

    # 8. 将调整后的图像粘贴到中心
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    a4_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # 9. 保存结果
    cv2.imwrite('restored_a4.jpg', a4_image)

print("原始图像尺寸:", img.shape)
print("检测到的轮廓数量:", len(contours))
print("校正后图像尺寸:", warped.shape if 'warped' in locals() else "未生成")
print("最终A4图像尺寸:", a4_image.shape)

