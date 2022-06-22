import numpy as np



def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0

def GazeTo3d(gaze):
    x = -np.cos(gaze[1]) * np.sin(gaze[0])
    y = -np.sin(gaze[1])
    z = -np.cos(gaze[1]) * np.cos(gaze[0])
    return np.array([x, y, z])


## 用numpy读列表计算视线方向角度误差(°)
## list形式为：XXXX.jpg pgaze0 pgaze1
##             XXXX.jpg pgaze0 pgaze1
## gaze gt file 文件名与图片同名后缀为：'.3dgdh'
## gaze gt file 内容为：gaze0 gaze1 gaze2 head0 head1 head2
def gazeErrorEye(dirsrc,prelist):

    infile = open(prelist)
    strinfo = infile.readlines()
    print('imgslen: ',len(strinfo))
    infile.close()

    errorsum = 0
    for i in range(len(strinfo)):
        
        if i % 100 == 0:
            print('dwi: ',i)

        gazestr = strinfo[i][:-1].split(' ')
        gaze2dp = []
        gaze2dp.append(float(gazestr[1]))
        gaze2dp.append(float(gazestr[2]))

        gaze3dp = GazeTo3d(gaze2dp)
        # print('gaze3d: ',gaze3d)

        gtfilestr = dirsrc + strinfo[i][:-4] + '3dgdh'
        gtfile = open(gtfilestr)
        gt3dgh = gtfile.readlines()
        gtfile.close()

        gt3dghsp = gt3dgh[0][:-1].split(' ')
        gaze3dgt = []
        gaze0 = float(gt3dghsp[0])
        gaze1 = float(gt3dghsp[1])
        gaze2 = float(gt3dghsp[2])
        gaze3dgt.append(gaze0)
        gaze3dgt.append(gaze1)
        gaze3dgt.append(gaze2)


        angleError = dot_product_angle(np.array(gaze3dp), np.array(gaze3dgt))
        # print('angleError: ',angleError)

        errorsum += angleError

    errorMean = errorsum/len(strinfo)
    print('errorMean: ',errorMean)




if __name__ == '__main__':


    ## 用numpy读列表计算视线方向角度误差(°)
    dirsrc = './MPIIGaze/'          ##文件路径
    prelist = './gaze2dpre.txt'     ##模型前向推理结果
    gazeErrorEye(dirsrc,prelist)