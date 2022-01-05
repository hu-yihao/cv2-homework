import cv2
import numpy as np
from numpy.linalg import norm

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class CardPredictor:
    def __init__(self):
        pass

    def img_first_pre(self, car_pic_file):
        """
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)  # 读取文件
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]  # 取彩色图片的高、宽
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            # 缩小图片
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # 关于interpolation 有几个参数可以选择:
        # cv2.INTER_AREA - 局部像素重采样，适合缩小图片。
        # cv2.INTER_CUBIC和 cv2.INTER_LINEAR 更适合放大图像，其中INTER_LINEAR为默认方法。

        img = cv2.GaussianBlur(img, (5, 5), 0)
        # 高斯滤波是一种线性平滑滤波，对于除去高斯噪声有很好的效果
        # 0 是指根据窗口大小（ 5,5 ）来计算高斯函数标准差

        oldimg = img
        # 转化成灰度图像
        # 转换颜色空间 cv2.cvtColor
        # BGR ---> Gray  cv2.COLOR_BGR2GRAY
        # BGR ---> HSV  cv2.COLOR_BGR2HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("tmp/img_gray.jpg", img)

        # ones()返回一个全1的n维数组
        Matrix = np.ones((20, 20), np.uint8)

        # 开运算:先进性腐蚀再进行膨胀就叫做开运算。它被用来去除噪声。 cv2.MORPH_OPEN
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)

        # 图片叠加与融合
        # g (x) = (1 − α)f0 (x) + αf1 (x)   a→（0，1）不同的a值可以实现不同的效果
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        # cv2.imwrite("tmp/img_opening.jpg", img_opening)
        # 创建20*20的元素为1的矩阵 开操作，并和img重合

        # Otsu’s二值化
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Canny 边缘检测
        # 较大的阈值2用于检测图像中明显的边缘  一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的
        # 较小的阈值1用于将这些间断的边缘连接起来
        img_edge = cv2.Canny(img_thresh, 100, 200)
        cv2.imwrite("tmp/img_edge.jpg", img_edge)

        Matrix = np.ones((4, 19), np.uint8)
        # 闭运算:先膨胀再腐蚀
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        # 开运算
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("tmp/img_xingtai.jpg", img_edge2)
        return img_edge2, oldimg








