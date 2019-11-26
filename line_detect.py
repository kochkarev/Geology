import math
import cv2
import numpy as np
import os
import glob
from utils import visualize_line_detection_result

def detect(input_path, output_path):
    
    images = glob.glob(os.path.join(input_path, "*.jpg"))

    origs = []
    edges = []
    origs_lines = []
    edges_lines = []

    for image in images:

        src = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  
        if src is None:
            print('Can not open {} file'.format(image))
            continue
        origs.append(cv2.cvtColor(src, cv2.COLOR_GRAY2RGB))

        thr_low = 20
        thr_high = 100
        canny = cv2.Canny(src, thr_low, thr_high, None, 3)
        edges.append(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))

        cdst1 = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        cdst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        cdstP1 = np.copy(cdst1)
        cdstP2 = np.copy(cdst2)

        # lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #         cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

        thr = 30
        min_len = 50
        max_gap = 10
        linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, thr, None, min_len, max_gap)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP1, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                cv2.line(cdstP2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        origs_lines.append(cv2.cvtColor(cdstP2, cv2.COLOR_BGR2RGB))
        edges_lines.append(cv2.cvtColor(cdstP1, cv2.COLOR_BGR2RGB))

    output_name = str(thr_low) + '_' + str(thr_high) + '_' + str(thr) + '_' + str(min_len)
    visualize_line_detection_result(origs, edges, origs_lines, edges_lines, len(origs), output_path, output_name)

    return 0
    
if __name__ == "__main__":
    
    input_path = os.path.join('input', 'dataset')
    output_path = os.path.join('output', 'lines')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    detect(input_path, output_path)