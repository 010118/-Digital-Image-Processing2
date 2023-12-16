from re import I, T
from cv2 import CASCADE_DO_CANNY_PRUNING
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray 
from skimage.util import img_as_ubyte
from PIL import Image
import cv2
from skimage import feature
from scipy.spatial import distance
from skimage.filters import laplace

def part1 ():
    I = io.imread("moon.png") #read image 
    # I = rgb2gray(I) #turn grey
    # I = img_as_ubyte(I) 

    h = I.shape[0] #Get the height of the input image I.
    w = I.shape[1]
   
    h_image,w_image = I.shape #Get the height and width of the input image I and assign them to h_image and w_image.
    gker = np.array([[0, -1, 0], [-1,4,-1],[0,-1,0]])  #Create a 3x3 numpy array gker representing the Laplacian filter kernel.
    h_ker,w_ker = gker.shape #  拿到kernal 的shape
   
    hf_ker = np.cast['int']((h_ker-1.)/2.) # Calculate the integer value of half of the height of the filter kernel gker and assign it to hf_ker
    wf_ker = np.cast['int']((w_ker-1.)/2.) # Calculate the integer value of half of the width of the filter kernel gker and assign it to wf_ker.
    
    padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) #这些代码行创建了两个 numpy 数组padding和output_padding，
                     #它们将分别用于存储填充后的输入图像和过滤后的输出图像。
    output_padding = np.zeros((h + hf_ker * 2 ,w + w_ker *2))

    padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = I # 把 原图放到padding里面

   
    for i in np.arange(hf_ker,h_image-hf_ker,1):
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding[i,j] += I[i+l,j+m]*gker[l+hf_ker,m+wf_ker]
#这部分代码是拉普拉斯滤波运算的核心。I它使用拉普拉斯核过滤输入图像gker并将过滤后的图像存储在output_padding数组中。
#这部分代码对输入图像和拉普拉斯核进行卷积运算，得到滤波后的输出图像
    plt.subplot(4, 2, 1) 
    plt.imshow(I, cmap='gray')
    plt.title("Original")

    temp = output_padding
    plt.subplot(4, 2, 2) 
    plt.imshow(output_padding, cmap='gray')
    plt.title("Laplace filterd image")

###################################################
    gker = np.array([[0,0,0,0,0], [0,1,0,1,0],[0,0,0,1,0]]) 

    h_ker,w_ker = gker.shape # 拿到kernal 的shape
   
    hf_ker = np.cast['int']((h_ker-1.)/2.) 
    wf_ker = np.cast['int']((w_ker-1.)/2.) 
    
    padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) 
    output_padding = np.zeros((h + hf_ker * 2 ,w + w_ker *2))

    padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = I # 把 原图放到padding里面

   
    for i in np.arange(hf_ker,h_image-hf_ker,1):
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding[i,j] += I[i+l,j+m]*gker[l+hf_ker,m+wf_ker]


    plt.subplot(4, 2, 3) 
    plt.imshow(I, cmap='gray')
    plt.title("Original")

    plt.subplot(4, 2, 4) 
    plt.imshow(output_padding, cmap='gray')
    plt.title(" filterd image ")

##########################################################
    gker = np.array([[0,0,0], [6,0,6],[0,0,0]]) 

    h_ker,w_ker = gker.shape # 拿到kernal 的shape
   
    hf_ker = np.cast['int']((h_ker-1.)/2.) # 算图片外壳的高
    wf_ker = np.cast['int']((w_ker-1.)/2.) # 算图片外壳的宽
    
    padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) 
    output_padding = np.zeros((h + hf_ker * 2 ,w + w_ker *2))

    padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = I # 把 原图放到padding里面

   
    for i in np.arange(hf_ker,h_image-hf_ker,1):
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding[i,j] += I[i+l,j+m]*gker[l+hf_ker,m+wf_ker]


    plt.subplot(4, 2, 5) 
    plt.imshow(I, cmap='gray')
    plt.title("Original")

    plt.subplot(4, 2, 6) 
    plt.imshow(output_padding, cmap='gray')
    plt.title(" filterd image ")
    
#############################################



    I = io.imread("moon.png") #读取图片
    # I = rgb2gray(I) #变成灰色
    # I = img_as_ubyte(I) 

    h = I.shape[0] 
    w = I.shape[1]
   
    h_image,w_image = I.shape 
    gker = np.array([[0, -1, 0], [-1,4,-1],[0,-1,0]]) 
    h_ker,w_ker = gker.shape # 拿到kernal 的shape
   
    hf_ker = np.cast['int']((h_ker-1.)/2.) # 算图片外壳的高
    wf_ker = np.cast['int']((w_ker-1.)/2.) # 算图片外壳的宽
    
    padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) 
    I = I.astype('float64')
    output_padding = np.zeros_like(I)

    padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = I # 把 原图放到padding里面

   
    for i in np.arange(hf_ker,h_image-hf_ker,1):
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding[i,j] += I[i+l,j+m]*gker[l+hf_ker,m+wf_ker]


    plt.subplot(4,2,7)
    plt.imshow((I).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Cameraman")
    plt.subplot(4,2,8)

    plt.imshow((I+output_padding).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Enhanced Cameraman")
    plt.show()


    

def part2():
    
    noisy = io.imread("noisy.jpg") #读取图片

    height = noisy.shape[0] # 图像的尺寸， 按照像素值计算，他的返回值为宽度和高度的二元组
    width = noisy.shape[1]


    temp_noisy = noisy
    #apply median filter to remove the noise
    image_processed_by_median = cv2.medianBlur(temp_noisy, 5)#apply the medianBlur filter
    #save the image  
    mean_image = "mean_image.jpg"
    cv2.imwrite(mean_image,image_processed_by_median)
 
   
    #Apply a Gaussian filter to the same noisy image.
    '''
    We should specify the width and height of the kernel which should be positive and odd.
    We also should specify the standard deviation in the X and Y directions, 
    sigmaX and sigmaY respectively. 
    If only sigmaX is specified, sigmaY is taken as the same as sigmaX. 
    If both are given as zeros, they are calculated from the kernel size. 
    Gaussian blurring is highly effective in removing Gaussian noise from an image.
    '''
    image_processed_by_Guassian = cv2.GaussianBlur(temp_noisy,(5,5),0) #在上面
    #save the image
    Gaussian_image = "Gaussian_image.jpg"
    cv2.imwrite(Gaussian_image,image_processed_by_Guassian)
    ###此代码使用 OpenCV 库将高斯模糊滤镜应用于图像“temp_noisy”。该函数cv2.GaussianBlur()采用三个参数：输入图像、核大小（在本例中为 5x5）和核的标准差（在本例中为 0，表示标准差是根据核大小自动计算的）。
    #然后使用函数将生成的模糊图像保存为名为“Gaussian_image.jpg”的新文件cv2.imwrite()。此函数有两个参数：将图像保存为的文件名和要保存的图像数据。在这种情况下，图像数据是函数的输出cv2.GaussianBlur()。

    #output three images
    plt.subplot(1, 3, 1) 
    plt.imshow(noisy, cmap='gray')
    plt.title("Original")
    
    plt.subplot(1, 3, 2) 
    plt.imshow(image_processed_by_median, cmap='gray')
    plt.title("median")

    plt.subplot(1, 3, 3) 
    plt.imshow(image_processed_by_Guassian, cmap='gray')
    plt.title("Gaussian")
    plt.show()


def part3():
    damage_camera = io.imread("damage_cameraman.png") #读取图片
    damage_mask = io.imread("damage_mask.png")
    temp_damage = io.imread("damage_cameraman.png")

    height = damage_camera.shape[0] # 图像的尺寸， 按照像素值计算，他的返回值为宽度和高度的二元组
    width =  damage_camera.shape[1]

    damage_pixel = [] #Create an empty list to store the pixel coordinates that need to be repaired.

    for i in range(height): # Iterate over the rows of the image.
      for j in range(width):# Iterate over the columns of the image.
        if damage_mask[i][j].all() == 0: #Check if the pixel at (i,j) in the mask image is damaged. If it is damaged (all channels are 0), then add its coordinates to the damage_pixel list.
          damage_pixel.append((i,j)) # If the pixel is damaged, append its coordinates to the damage_pixel list.
    
    i = 1  #Initialize the variable i to 1.
    while(i<= 5000): #  Loop 5000 times.
      Gaussian_image = cv2.GaussianBlur(temp_damage,(5,5),0)#J = GaussianSmooth(J)  Smooth damaged image
      #Apply a Gaussian filter to the damaged image using cv2.GaussianBlur() function. The (5,5) argument specifies the size of the kernel, and 0 specifies the standard deviation of the filter along both axes.
      for j in damage_pixel: #Iterate over the coordinates of the damaged pixels.遍历损坏像素的坐标。
        temp_damage[j] = Gaussian_image[j] #J(U) = I(U) (Copy good pixels Iterate over the coordinates of the damaged pixels.)
        #将平滑（修复）图像中的相应像素复制到损坏的图像中。
      i = i + 1 #Increment the loop counter.增加循环计数器
            
    plt.subplot(1, 2, 1) 
    plt.imshow(damage_camera, cmap='gray')
    plt.title("damaged image")

    plt.subplot(1, 2, 2) 
    plt.imshow(temp_damage, cmap='gray')
    plt.title("restored image")

    plt.show()


def part4():

    ex2 = io.imread("ex2.jpg") #读取图片 input

    height = ex2.shape[0] # 图像的尺寸， 按照像素值计算，他的返回值为宽度和高度的二元组
    width =  ex2.shape[1] 

    h_image,w_image = ex2.shape 
    gker = np.array([[-1, 0, 1], [-2,0,2],[-1,0,1]]) #input kernal
    h_ker,w_ker = gker.shape # 拿到kernal 的shape
   

    hf_ker = np.cast['int']((h_ker-1.)/2.) # 算padding的高和宽的大小 
    wf_ker = np.cast['int']((w_ker-1.)/2.) # 
    

    padding = np.zeros((height + hf_ker * 2 , width + w_ker *2 )) 
    output_padding = np.zeros((height + hf_ker * 2 ,width + w_ker *2))

    padding[hf_ker:(height+hf_ker), wf_ker:(width+wf_ker)] = ex2 # 把 原图放到padding里面 （预处理）

    for i in np.arange(hf_ker,h_image-hf_ker,1):    #把 kernal不停移动直到算完整个图片 
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding[i,j] += ex2[i+l,j+m]*gker[l+hf_ker,m+wf_ker] #把后面的存到前面去

    h_image,w_image = ex2.shape #unpacks the dimensions of the input image ex2 and assigns them to the variables h_image and w_image respectively, 
                                #which represent the height and width of the image in pixels.
    gker_vertical = np.array([[1, 2, 1], [0,0,0],[-1,-2,-1]]) #input kernal
    h_ker,w_ker = gker.shape # 拿到kernal 的shape
   

    hf_ker = np.cast['int']((h_ker-1.)/2.) # 算padding的高和宽的大小 
    wf_ker = np.cast['int']((w_ker-1.)/2.) # 
    
    padding = np.zeros((height + hf_ker * 2 , width + w_ker *2 )) 
    output_padding_vertical = np.zeros((height + hf_ker * 2 ,width + w_ker *2))

    padding[hf_ker:(height+hf_ker), wf_ker:(width+wf_ker)] = ex2 # 把 原图放到padding里面 （预处理）

    for i in np.arange(hf_ker,h_image-hf_ker,1):    #把 kernal不停移动直到算完整个图片 
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding_vertical[i,j] += ex2[i+l,j+m]*gker[l+hf_ker,m+wf_ker] #把后面的存到前面去


    E = np.sqrt(output_padding**2 + output_padding_vertical**2) # Edge strength or Gradient magnitude
 #通过取水平和垂直梯度平方和的平方根来计算每个像素处梯度向量的大小。这导致图像中较亮的像素对应于原始图像中较强的边缘。
#calculates the magnitude of the gradient vector at each pixel by taking the square root of the sum of the squared horizontal and vertical gradients. 
# This results in an image where the brighter pixels correspond to the stronger edges in the original image.

    plt.figure(figsize=(15,7))
    plt.subplot(4, 1, 1) 
    plt.imshow(ex2, cmap='gray')
    plt.title("image")

    plt.subplot(4, 1, 2) 
    plt.imshow(output_padding, cmap='gray')
    plt.title("Horizontal")

    plt.subplot(4, 1, 3 ) 
    plt.imshow(output_padding_vertical, cmap='gray')
    plt.title("Vertical")

    plt.subplot(4, 1, 4 ) 
    plt.imshow(E, cmap='gray')
    plt.title("Gradient")

    plt.show()


def part5():

    picture = io.imread("ex2.jpg", as_gray=True) #读取图片 input
    target = io.imread("canny_target.jpg", as_gray=True) 
    best_distance = 100000 #  best_distance 初始化为一个较大的数字，例如 1e10
    best_params = [0,0,0]   #用零初始化 best_params 数组。
    for low_thresh in range(50,100, 20):
        for high_threshold in range(100,200,15):
            for sigma in np.arange(1.0,3.0,0.2):
                canny = feature.canny(image=picture, sigma=sigma, low_threshold=low_thresh, high_threshold=high_threshold) 
                # Apply the Canny method and the parameters to the image
                this_dist = distance.cosine(canny.flatten(),target.flatten())#公式+变成向量
                if this_dist < best_distance:
                    best_distance = this_dist
                    best_params = [sigma,low_thresh,high_threshold]
    my_image = feature.canny(image = picture, sigma=best_params[0],low_threshold=best_params[1],high_threshold=best_params[2])
    #feature.canny is a function that implements the Canny edge detection algorithm
    print(best_distance)
    print(best_params)


    image_processed_by_Guassian = cv2.GaussianBlur(picture,(5,5),0) # This code applies a Gaussian filter to the original image picture using the OpenCV GaussianBlur() function, which takes the image and the size of the kernel as input. 
                                #(5,5) in this case is the size of the Gaussian kernel used for blurring the image. 
                                #The 0 parameter is the standard deviation of the Gaussian kernel, indicating that it is auto-calculated based on the kernel size.
    #save the image
    Gaussian_image = "Gaussian_image.jpg"
    cv2.imwrite(Gaussian_image,image_processed_by_Guassian)
      
                
    plt.subplot(4, 1, 1) 
    plt.imshow(picture, cmap='gray')
    plt.title("image")

    plt.subplot(4, 1, 2) 
    plt.imshow(image_processed_by_Guassian , cmap='gray')
    plt.title("Guassian")


    plt.subplot(4, 1, 3) 
    plt.imshow(target, cmap='gray')
    plt.title("target")

    plt.subplot(4, 1, 4) 
    plt.imshow(my_image, cmap='gray')
    plt.title("my_image")


    plt.show()



if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()



