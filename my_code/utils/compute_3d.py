import numpy as np

def compute_unit_vector_3d(p1, p2):
    """
    计算从点 p1 指向 p2 的单位方向矢量

    参数：
        p1, p2: 3D 坐标，形如 [x, y, z] 的列表或 numpy 数组

    返回：
        单位向量（numpy 数组）
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("两点重合，无法计算单位方向矢量。")
    return vec / norm

def compute_circle_center_3d(P1, P2, P3):
    P1, P2, P3 = np.array(P1), np.array(P2), np.array(P3)

    # 向量 u, v 定义圆所在的平面
    u = P2 - P1
    v = P3 - P1

    # 平面的法向量
    n = np.cross(u, v)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("Points are colinear; cannot determine unique circle")

    # 单位法向量
    n = n / n_norm

    # 计算两个中点
    mid_ab = (P1 + P2) / 2
    mid_ac = (P1 + P3) / 2

    # 中垂线方向 = u 的垂直向量（在平面内）
    perp_ab = np.cross(n, u)
    perp_ac = np.cross(n, v)

    # 求两条中垂线交点
    def line_intersection(p1, d1, p2, d2):
        A = np.array([d1, -d2]).T
        b = p2 - p1
        t = np.linalg.lstsq(A, b, rcond=None)[0]
        return p1 + t[0] * d1

    center = line_intersection(mid_ab, perp_ab, mid_ac, perp_ac)
    return center

def move_along_vector(point, unit_vector, distance):
    """
    沿单位方向矢量移动指定距离，返回移动后的点坐标

    参数：
        point: 初始点坐标，形如 [x, y, z]
        unit_vector: 单位方向矢量，形如 [x, y, z]
        distance: 沿该方向移动的距离（float）

    返回：
        新的点坐标（numpy 数组）
    """
    point = np.array(point)
    unit_vector = np.array(unit_vector)

    # 验证 unit_vector 是否为单位长度
    if not np.isclose(np.linalg.norm(unit_vector), 1.0):
        raise ValueError("输入的向量不是单位矢量，请先归一化。")

    new_point = point + unit_vector * distance
    return new_point


def direction_error(v1, v2, degrees=True):
    """
    计算两个三维向量的夹角

    参数：
        v1, v2: 3D 向量，形如 [x, y, z] 的列表或 numpy 数组
        degrees: 是否将结果转换为角度（默认为 True）

    返回：
        夹角（弧度或角度）
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    # 归一化
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 点积并限制在 [-1, 1]，避免浮点误差
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)

    # 计算夹角
    angle = np.arccos(cos_theta)

    if degrees:
        angle = np.degrees(angle)

    return angle