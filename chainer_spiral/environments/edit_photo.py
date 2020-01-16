import cv2
import numpy as np
import math
import sys
try:
    from .dehaze.src import dehaze
except:
    from dehaze.src import dehaze


def nothing(x):
    pass

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_inverse(y):
    epsilon = 10**(-3)
    y_ = y.copy()
    y_ = relu(y_-epsilon)+epsilon
    y_ = 1-epsilon-relu((1-epsilon)-y_)
    return -np.log(1/(y_)-1)

def relu(x):
    x_ = x.copy()
    x_[x_<0] = 0
    return x_

class Sigmoid():
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_image, parameters):
        return [sigmoid(list_image[0])]

class SigmoidInverse():
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_image, parameters):
        return [sigmoid_inverse(list_image[0])]

class AdjustContrast():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["contrast"]

    def __call__(self, list_editted, parameters):
        editted = list_editted[0]
        mean = editted.mean()
        editted_ = (editted-mean)*(parameters[0]+1)+mean
        editted_ = relu(editted_)
        editted_ = 1-relu(1-editted_)
        return [editted_]

class AdjustDehaze():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["dehaze"]

    def __call__(self, list_editted, parameters):
        editted = list_editted[0]
        scale = max((editted.shape[:2])) / 512.0
        omega = parameters[0]
        editted_ = dehaze.DarkPriorChannelDehaze(
            wsize=int(15*scale), radius=int(80*scale), omega=omega,
            t_min=0.25, refine=True)(editted * 255.0) / 255.0
        editted_ = relu(editted_)
        editted_ = 1-relu(1-editted_)
        return [editted_]

class AdjustClarity():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["clarity"]

    def __call__(self, list_editted, parameters):
        editted = list_editted[0]
        scale = max((editted.shape[:2])) / 512.0
        clarity = parameters[0]

        unsharped = cv2.bilateralFilter((editted*255.0).astype(np.uint8),
                                            int(32*scale), 50, 10*scale)/255.0
        editted_ = editted + (editted-unsharped) * clarity
        return [editted_]

class AdjustExposure():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["exposure"]

    def __call__(self, list_sigmoid_inversed, parameters):
        exposure = parameters[0]
        return [list_sigmoid_inversed[0] + exposure*5]

class AdjustTemp():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["temp"]

    def __call__(self, list_sigmoid_inversed, parameters):
        temp = parameters[0]
        sigmoid_inversed_ = list_sigmoid_inversed[0].copy()
        if temp > 0:
            sigmoid_inversed_[:,:,1] += temp*1.6
            sigmoid_inversed_[:,:,2] += temp*2
        else:
            sigmoid_inversed_[:,:,0] -= temp*2.0
            sigmoid_inversed_[:,:,1] -= temp*1.0
        return [sigmoid_inversed_]


class AdjustTint():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["tint"]

    def __call__(self, list_sigmoid_inversed, parameters):
        tint = parameters[0]
        sigmoid_inversed_ = list_sigmoid_inversed[0].copy()
        if tint > 0:
            sigmoid_inversed_[:,:,0] += tint*2
            sigmoid_inversed_[:,:,2] += tint*1
        else:
            sigmoid_inversed_[:,:,1] -= tint*2
            sigmoid_inversed_[:,:,2] -= tint*1
        return [sigmoid_inversed_]

class AdjustShadows():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["shadows"]

    def __call__(self, list_hsv, parameters):
        shadows = parameters[0]
        v = list_hsv[2]
        shadows_mask = 1-sigmoid((v-0)*5)
        return [list_hsv[0], list_hsv[1], v*(1+shadows_mask*shadows*5)]

class AdjustHighlights():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["highlights"]

    def __call__(self, list_hsv, parameters):
        hilights = parameters[0]
        v = list_hsv[2]
        hilights_mask = sigmoid((v-1)*5)
        return [list_hsv[0], list_hsv[1], 1-(1-v)*(1-hilights_mask*hilights*5)]

class AdjustBlacks():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["blacks"]

    def __call__(self, list_hsv, parameters):
        blacks = parameters[0]+1
        v = list_hsv[2]
        return [list_hsv[0], list_hsv[1], v+(1-v)*(math.sqrt(blacks)-1)*0.2]

class AdjustWhites():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["whites"]

    def __call__(self, list_hsv, parameters):
        whites = parameters[0]+1
        v = list_hsv[2]
        return [list_hsv[0], list_hsv[1], v+(v)*(math.sqrt(whites)-1)*0.2]

class Bgr2Hsv():
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_editted, parameters):
        editted = list_editted[0]

        max_bgr = editted.max(axis=2)
        min_bgr = editted.min(axis=2)

        b_g = editted[:,:,0]-editted[:,:,1]
        g_r = editted[:,:,1]-editted[:,:,2]
        r_b = editted[:,:,2]-editted[:,:,0]

        b_min_flg = (1-relu(np.sign(b_g)))*relu(np.sign(r_b))
        g_min_flg = (1-relu(np.sign(g_r)))*relu(np.sign(b_g))
        r_min_flg = (1-relu(np.sign(r_b)))*relu(np.sign(g_r))

        epsilon = 10**(-5)
        h1 = 60*g_r/(max_bgr-min_bgr+epsilon)+60
        h2 = 60*b_g/(max_bgr-min_bgr+epsilon)+180
        h3 = 60*r_b/(max_bgr-min_bgr+epsilon)+300
        h = h1*b_min_flg + h2*r_min_flg + h3*g_min_flg

        v = max_bgr
        s = (max_bgr-min_bgr)/(max_bgr+epsilon)

        return [h,s,v]

class AdjustVibrance():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["vibrance"]

    def __call__(self, list_hsv, parameters):
        vibrance = parameters[0]+1
        s = list_hsv[1]
        # vibrance_flg = np.sign(relu(0.5-s))
        vibrance_flg = -sigmoid((s-0.5)*10) + 1
        return [list_hsv[0], s*vibrance*vibrance_flg + s*(1-vibrance_flg), list_hsv[2]]

class AdjustSaturation():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["saturation"]

    def __call__(self, list_hsv, parameters):
        saturation = parameters[0]+1
        s = list_hsv[1]
        s_ = s*saturation
        s_ = relu(s_)
        s_ = 1-relu(1-s_)
        return [list_hsv[0], s_, list_hsv[2]]

class Hsv2Bgr():
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_hsv, parameters):
        h,s,v = list_hsv
        h = h*relu(np.sign(h-0))*(1-relu(np.sign(h-360))) + (h-360)*relu(np.sign(h-360))*(1-relu(np.sign(h-720)))\
                + (h+360)*relu(np.sign(h+360))*(1-relu(np.sign(h-0)))
        h60_flg = relu(np.sign(h-0))*(1-relu(np.sign(h-60)))
        h120_flg = relu(np.sign(h-60))*(1-relu(np.sign(h-120)))
        h180_flg = relu(np.sign(h-120))*(1-relu(np.sign(h-180)))
        h240_flg = relu(np.sign(h-180))*(1-relu(np.sign(h-240)))
        h300_flg = relu(np.sign(h-240))*(1-relu(np.sign(h-300)))
        h360_flg = relu(np.sign(h-300))*(1-relu(np.sign(h-360)))

        C = v*s
        b = v-C + C*(h240_flg+h300_flg) + C*((h/60-2)*h180_flg + (6-h/60)*h360_flg)
        g = v-C + C*(h120_flg+h180_flg) + C*((h/60)*h60_flg + (4-h/60)*h240_flg)
        r = v-C + C*(h60_flg+h360_flg) + C*((h/60-4)*h300_flg + (2-h/60)*h120_flg)

        return [np.concatenate([np.expand_dims(b, axis=2),np.expand_dims(g, axis=2),np.expand_dims(r, axis=2)], axis=2)]

class Srgb2Photopro():
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_srgb, parameters):
        srgb = list_srgb[0]
        k=0.055
        thre_srgb = 0.04045
        a = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
        b = np.array([[1.3459433,-0.2556075,-0.0511118],[-0.5445989,1.5081673,0.0205351],[0.0000000,0.0000000,1.2118128]])
        M = b.dot(a)
        M = M/M.sum(axis=1).reshape((-1,1))
        thre_photopro = 1/512.0

        srgb[srgb<=thre_srgb] /= 12.92
        srgb[srgb>thre_srgb] = ((srgb[srgb>thre_srgb]+k)/(1+k))**2.4

        image = srgb
        sb = image[:,:,0:1]
        sg = image[:,:,1:2]
        sr = image[:,:,2:3]
        photopror = sr*M[0][0]+sg*M[0][1]+sb*M[0][2]
        photoprog = sr*M[1][0]+sg*M[1][1]+sb*M[1][2]
        photoprob = sr*M[2][0]+sg*M[2][1]+sb*M[2][2]

        photopro = np.concatenate((photoprob,photoprog,photopror),axis=2)
        photopro = np.clip(photopro,0,1)
        photopro[photopro>=thre_photopro] = photopro[photopro>=thre_photopro]**(1/1.8)
        photopro[photopro<thre_photopro] *= 16

        return [photopro]


class Photopro2Srgb():
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_photopro, parameters):
        photopro = list_photopro[0]
        thre_photopro = 1/512.0*16

        a = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
        b = np.array([[1.3459433,-0.2556075,-0.0511118],[-0.5445989,1.5081673,0.0205351],[0.0000000,0.0000000,1.2118128]])
        M = b.dot(a)
        M = M/M.sum(axis=1).reshape((-1,1))
        M = np.linalg.inv(M)
        k=0.055
        thre_srgb = 0.04045/12.92

        photopro[photopro<thre_photopro] *= 1.0/16
        photopro[photopro>=thre_photopro] = photopro[photopro>=thre_photopro]**(1.8)

        photoprob = photopro[:,:,0:1]
        photoprog = photopro[:,:,1:2]
        photopror = photopro[:,:,2:3]
        sr = photopror*M[0][0]+photoprog*M[0][1]+photoprob*M[0][2]
        sg = photopror*M[1][0]+photoprog*M[1][1]+photoprob*M[1][2]
        sb = photopror*M[2][0]+photoprog*M[2][1]+photoprob*M[2][2]

        srgb = np.concatenate((sb,sg,sr),axis=2)

        srgb = np.clip(srgb,0,1)
        srgb[srgb>thre_srgb] = (1+k)*srgb[srgb>thre_srgb]**(1/2.4)-k
        srgb[srgb<=thre_srgb] *= 12.92

        return [srgb]


class PhotoEditor():
    def __init__(self):
        self.edit_funcs = [Srgb2Photopro(), AdjustDehaze(), AdjustClarity(), AdjustContrast(),
                SigmoidInverse(), AdjustExposure(), AdjustTemp(), AdjustTint(),
                Sigmoid(), Bgr2Hsv(), AdjustWhites(), AdjustBlacks(), AdjustHighlights(),
                AdjustShadows(), AdjustVibrance(), AdjustSaturation(), Hsv2Bgr(), Photopro2Srgb()]

        self.num_parameters = 0
        for edit_func in self.edit_funcs:
            self.num_parameters += edit_func.num_parameters

    def __call__(self, photo, parameters):
        output_list = [photo]
        num_parameters = 0
        for edit_func in self.edit_funcs:
            output_list = edit_func(output_list,
                parameters[num_parameters : num_parameters + edit_func.num_parameters])
            num_parameters = num_parameters + edit_func.num_parameters

        return output_list[0]



def edit_demo(photo, parameters_=None):
    cv2.namedWindow('photo', cv2.WINDOW_NORMAL)
    cv2.namedWindow('parameter', cv2.WINDOW_NORMAL)

    parameter_dammy = np.ones((1,400,3)) * (233 / 255)

    input = photo / 255.0
    photo_editor = PhotoEditor()

    if parameters_ is None:
        parameters = np.zeros(photo_editor.num_parameters)
    else:
        parameters = parameters_.copy()

    j = 0
    for edit_func in photo_editor.edit_funcs:
        for i in range(edit_func.num_parameters):
            cv2.createTrackbar(edit_func.slider_names[i],edit_func.window_names[i],int((parameters[j]+1)*100),200,nothing)
            j += 1

    print("[[Press esc to quit.]]")
    while(1):
        parameters = []
        for edit_func in photo_editor.edit_funcs:
            for i in range(edit_func.num_parameters):
                parameters.append(cv2.getTrackbarPos(edit_func.slider_names[i],edit_func.window_names[i])/100.0-1)

        output = photo_editor(input.copy(), parameters)
        cv2.imshow('photo', np.hstack([input, output]))
        cv2.imshow('parameter',parameter_dammy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__=="__main__":
    parameters = np.array([0.125, 0.125, 0.375, 0.125, 0., 0.0625, 0.9375, 0.375, 0.0625, 0., 0.125, 0.125])
    image = cv2.imread("../../fivek_dataset/original/a0676-kme_609.tif")
    edit_demo(image*1.0, parameters)
