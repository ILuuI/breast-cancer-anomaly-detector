"""
Supresion de No Maximos (NMS) usada para consolidar detecciones
redundantes generadas por el barrido de ventana deslizante de la
via Micro.
"""
import numpy as np

def nms_fast(boxes, confs, thresh=0.2):
    if len(boxes) == 0: return [], []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confs)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]; pick.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[idxs[:last]]), np.maximum(y1[i], y1[idxs[:last]])
        xx2, yy2 = np.minimum(x2[i], x2[idxs[:last]]), np.minimum(y2[i], y2[idxs[:last]])
        w_, h_ = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        ov = (w_ * h_) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(ov > thresh)[0])))
    return boxes[pick].astype("int"), confs[pick]
