import cv2
import numpy as np

Min_Area = 2000  # 车牌区域允许最大面积

"""
    该文件包含读文件函数
    取零值函数
    矩阵校正函数
    颜色判断函数
"""


def img_read(filename):
    '''
        以uint8方式读取filename
        放入imdecode中，cv2.IMREAD_COLOR读取彩色照片
    '''

    # cv2.IMREAD_COLOR：读入一副彩色图像。图像的透明度会被忽略，这是默认参数
    # cv2.IMREAD_GRAYSCALE：以灰度模式读入图像
    # cv2.IMREAD_UNCHANGED：读入一幅图像，并且包括图像的 alpha 通道
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    row_num_limit = 21
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl

def img_findContours(img_contours):
    # 查找轮廓
    # cv2.findContours()
    # 有三个参数，第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法。参数为二值图，即黑白的（不是灰度图）
    # 返回值有三个，第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构。
    # 轮廓（第二个返回值）是一个 Python列表，其中存储这图像中的所有轮廓。
    # 每一个轮廓都是一个 Numpy 数组，包含对象边界点（x，y）的坐标。
    contours, hierarchy = cv2.findContours(img_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_TREE建立一个等级树结构的轮廓
    #  cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，
    #  只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

    # cv2.contourArea计算该轮廓的面积
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
    # print("findContours len = ", len(contours))

    # 面积小的都筛选掉
    car_contours = []
    for cnt in contours:
        ant = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        width, height = ant[1]
        if width < height:
            width, height = height, width
        ration = width / height

        if 2 < ration < 5.5:
            car_contours.append(ant)
            # box = cv2.boxPoints(ant) # 获得要绘制这个矩形的 4 个角点

    return car_contours

