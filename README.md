## EX.NO: 06 <br>
## DATE: 06.04.2023
## <p align="center">IMPLEMENTATION OF FILTERS</p>

## Aim:

To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:

Anaconda - Python 3.7

## Algorithm:

### Step1
Import the necessary modules.

### Step2
For performing smoothing operation on a image. 
- Average filter
```python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```
- Weighted average filter
```python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```
- Gaussian Blur 
```python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
```
- Median filter
```python
median=cv2.medianBlur(image2,13)
```

### Step3
For performing sharpening on a image.
- Laplacian Kernel
```python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```
- Laplacian Operator
```python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```

### Step4
Display all the images with their respective filters.

## Program:
### Developed By   : P S Chetan
### Register Number: 212220230033
```PYTHON
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("simp.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
```

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```

ii) Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```

iii) Using Gaussian Filter
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```

iv) Using Median Filter
```Python
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```

ii) Using Laplacian Operator
```Python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters
i) Using Averaging Filter

![image](https://user-images.githubusercontent.com/74660507/167992682-a28327f5-12b7-4c01-a8dd-034a9ffaf803.png)

ii) Using Weighted Averaging Filter

![image](https://user-images.githubusercontent.com/74660507/167992717-4c10f23c-95f3-4de6-8c13-75634cdf0d36.png)

iii) Using Gaussian Filter

![image](https://user-images.githubusercontent.com/74660507/167992759-fdf5a3a7-013e-4d1f-9eb6-003274651bd1.png)

iv) Using Median Filter

![image](https://user-images.githubusercontent.com/74660507/167992794-af5e5a51-7720-41e6-a3c5-3c2d3f1948da.png)


### 2. Sharpening Filters
i) Using Laplacian Kernal

![image](https://user-images.githubusercontent.com/74660507/167992873-2bbb44a7-b0a3-4cb5-92a7-30f6619ba5c5.png)

ii) Using Laplacian Operator


![image](https://user-images.githubusercontent.com/74660507/167992902-3e54210b-6bd1-4c98-a5be-bcd046e86175.png)



## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
