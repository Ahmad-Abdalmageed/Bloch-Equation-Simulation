# Bloch Equation Simulation 

### An implementation of Bloch equations that describes the behavior of the bulk magnetization 

Main importations


```python
import bloch as b  # a Class which is implemented in it`s own module 
import numpy as np
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import matplotlib.pyplot as plt
%matplotlib notebook
%matplotlib notebook
```

##### instantiation of the main class responsible for the calculations needed and printing the accompanied doc string 

###### the file can be tracked here [Bloch](bloch.py)


```python
m = b.magentization(900*10**-3, 50*10**-3, 1.5)
print(b.magentization.__doc__)
```


        Responsible for calculating the magnetization vector.
        Implements the following:
        * calculate the magnetization vector after application of Mo [0 0 Mo]
        * Returns the vector into its relaxation state



#### we searched for T1 and T2 values for different tissues and found many results 

##### the values we tried are from this [link](https://mri-q.com/why-is-t1--t2.html)

![T1T2](https://mri-q.com/uploads/3/4/5/7/34572113/5092979_orig.gif)

#### applied an RF pulse for 1 sec 


```python
m.rotate(1)
print(m.rotate.__doc__)
```


            Rotates the magnetization vector by application of an RF pulse for a given time t
            ================== =================================================
            **Parameters**
            t                  Time in seconds
            ================== =================================================



## plotting 

### The following chunk of code is responsible for making an animation of the bulk magnetization\`s trajectory 

- using matplotlib funcAnimation and quiver for 3d plotting 
- the plot is initialized with the first values returned from the rotations of the vector 
- an update function is given for FuncAnimation which updates the plot\`s data with the next value to show in the next frame 


```python
fig = plt.figure()
ax = fig.gca(projection='3d')

# Origin
x, y, z = (0, 0, 0)

# Directions of the vector 
u = m.vector[0, 0]  # x Component 
v = m.vector[0, 1]  # y Component
w = m.vector[0, 2]   # z Component 

quiver = ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="red")
ax.plot(m.vector[:0, 0], m.vector[:0, 1], m.vector[:0, 2], color='r', label="Trajectory")

def update(t):
    global quiver
    u = m.vector[t, 0]
    v = m.vector[t, 1]
    w = m.vector[t, 2]
    quiver.remove()
    quiver= ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1)
    ax.plot(m.vector[:t, 0], m.vector[:t, 1], m.vector[:t, 2], color='r', label="Trajectory")
    
ax.set_xlim3d([-0.3, 0.3])
ax.set_xlabel('X')

ax.set_ylim3d([-0.3, 0.3])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.5, 1.5])
ax.set_zlabel('Z')

ax.view_init(elev= 0.9, azim=-45)
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200, blit= True)
ax.legend()
ani.save("magnetization.gif")
plt.show()
```

##### Note : the animation is interactive on Jupyter

### the [Animation](magnetization.gif)



​	![](magnetization.gif)



### Third Part 

#### Applying Fourier Transform on an image 

###### importing a class made for the [image's](image.py) loading and performing Fourier transform 


```python
import image  # a class for image`s processes 
```


```python
imageSlice = image.image()
print(image.image().__doc__)
```


        Responsible for all interactions with images.
        Implements the following:
        * Loading the image data to the class
        * Apply Fourier Transformation to the image
        * Extract the following components from the transformations :
            - Real Component
            - Imaginary Component
            - Phase
            - Magnitude




```python
imageSlice.loadImage("78146.png", greyScale=False)
print(imageSlice.loadImage.__doc__)
```

    the image loaded shape is  (230, 230, 3)
    
            Implements the following:
            * Loading the image from specified path
            * Normalize the image values
            ================== =============================================================================
            **Parameters**
            Path               a string specifying the absolute path to image, if provided loads this image
                               to the class`s data
            data               numpy array if provided loads this data directly
            fourier            numpy array if provided loads the transformed data
            imageShape         a tuple of ints identifying the image shape if any method is used except using
                               path
            greyScale          if True the image is transformed to greyscale via OpenCV`s convert image tool
            ================== =============================================================================

## UPDATE 

##### plotted a new image and visualized the K-space components


```python
import matplotlib.cm as cm
```


```python
fig2 = plt.figure()
plt.title("Loaded Image/ Ankle")
plt.axis("off")
plt.imshow(imageSlice.imageData, cmap=cm.gray)
```

![](images/leg.jpg)


```python
imageSlice.fourierTransform()
fig3 = plt.figure()
plt.title("Real K-Space Component")
plt.ylabel("Ky")
plt.xlabel("Kx")
plt.imshow(imageSlice.realComponent(logScale=True))
```

![](images/8.png)

```python
fig4 = plt.figure()
plt.title("Imaginary K-Space Component")
plt.ylabel("Ky")
plt.xlabel("Kx")
plt.imshow(imageSlice.phase(), cmap=cm.gray)
```



![](images/4.png)


```python
print("Function`s description")
print("imageSlice.fourierTransform: ")
print(imageSlice.fourierTransform.__doc__)
print("imageSlice.magnitude:")
print(imageSlice.magnitude.__doc__)
print("imageSlice.phase: ")
print(imageSlice.phase.__doc__)
```

    Function`s description
    imageSlice.fourierTransform: 
    
            Applies Fourier Transform on the data of the image and save it in the specified attribute
            ================== ===========================================================================
            **Parameters**
            shifted            If True will also apply the shifted Fourier Transform
            ================== ===========================================================================
            
    imageSlice.magnitude:
    
            Extracts the image`s Magnitude Spectrum from the image`s Fourier data
            ================== ===========================================================================
            **Parameters**
            LodScale           If True returns 20 * np.log(ImageFourier)
            ================== ===========================================================================
            **Returns**
            array              a numpy array of the extracted data
            ================== ===========================================================================
            
    imageSlice.phase: 
    
            Extracts the image`s Phase Spectrum from the image`s Fourier data
            ================== ===========================================================================
            **Parameters**
            shifted           If true applies a phase shift on the returned data
            ================== ===========================================================================
            **Returns**
            array              a numpy array of the extracted data
            ================== ===========================================================================


​    



### Fourth Part 

#### Visualizing the Field\`s in-uniformity  


```python
field = 3.0  # Tesla 
delta = 0.5
```


```python
Bz = np.random.uniform(field-delta, field+delta, size=10)
```


```python
fig5 = plt.figure()
plt.title("Magnetic Field`s Randomality")
plt.xlabel("Measured Point")
plt.ylabel("The Measured Field")
plt.hlines(3,0, 10, label="The Field Value")
plt.scatter(range(0, 10), Bz, label="Different Measured points")
plt.legend()
```

![](images/6.png)

# Second Assignment

## Visualizing the differences in Angular Frequencies 


```python
G = 42.6  # for water molecules 
omega = G* Bz
```


```python
fig6 = plt.figure()
plt.title("Angular Frequencies")
plt.xlabel("Measured Point")
plt.ylabel("Frequency(MHZ)")
plt.hlines(3*G,0, 10, label="The Field Value")
plt.scatter(range(0, 10), omega, label="Different Measured Frequencies")
plt.legend()
plt.show()
```

<img src="data:image/png;base64,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qIFevXopHxd/MZqp+5sKsDNnzkCj0Rid+Xr55ZfRoEED5eOYmBgIIZCenm5w3MyZM+Hr62twxvPWrVuoXr063nnnHYP5dejQweC+EyZMQLVq1ZSPy3LpQk5ODiZPnozu3bsDuB9/+nkXD90hQ4YgNDTU4JKLCxcuwNnZGevXrzf5PDqdDmpgO38AACAASURBVCdOnIAQAj/88AOAB9c/37x50+R9zA3dd9991+C4lJQUCCFw4MABg9t79+6NHj16KPOqVasWhg8fbnDMP/7xjzK/GK3o92rw4MEmH2f69OmoWrUq/vrrL+W2mzdvws/PDwsXLgRw/+ymEMLge5qXl4fq1aubHboFBQUIDAw0+jrXr18PJycn5eeAPnRXr16tHHP37l34+flh+vTpym3t2rVDq1atDKK+qOKh+8svv0AIgU8//dTguH/+859o0qQJAODy5csQQii/4SgNhi6RbWDoyjF0LUSN0E04ftFk4BYfCcctt/AfFroBAQHlErorVqyAVqtFTk6Owdm9LVu2QAihXAccExODgIAAo3l07NgRzz//vMF98/Pz8cwzz+Af//iHwfzefPNNg/uuXLkSQgjlutSyhu73338PBwcH/PHHH3j88ccxa9YsAMahW6NGDUyePNlori1btsTEiROV43bu3IkOHTqgSpUqBiGoP8N9+vRpODg4IDIyEjt27DA4cy37PgPy0C2+08H06dNRvXp1o3m+++678PHxAXB/nQshsGPHDoP7fvbZZ6UO3XfffRfp6enKyMzMVI4ZPHgwAgMDje7btm1bvPDCC0Zz6927N55//nkAwMcff2zw56r34osvmh26+n9k7Nmzx+D59HGZlJQE4EHoFr8MpV27dhg8eDCA+7810Gg0+OCDD6Tfm+Khu3TpUjg4OODmzZsGz79x40YIIXDr1i3odDoEBQWhRYsW+PTTT0v1FyJDl8g2MHTlGLoWokboDlt7VHo2t+hZ3eHrvnuk5ylJSaGbl5cHR0dHDBs2TLmtrKH79ttvS7eaKvoXfkxMDFq2bGk0l9DQUOl9i/66v/j8AMNYBcoeugDQsGFDTJo0CRqNBhkZGQCMQ9fR0VE61379+gEAjh49CgcHB0RHRyMxMRFpaWlK/Kxbt055rD179qBz585wcHCAk5OTwbW+5obutWvXDI576aWXSvwz+fvvv5GWlqZc31vU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c8gxdC2Ei46IiIjUwOaQY+haCBcdERERqYHNIcfQtRAuOiIiIlIDm0OOoWshXHRERESkBjaHHEPXQrjoiIiISA1sDjmGroVw0REREZEa2BxyDF0L4aIjIiIiNbA55Bi6FsJFR0RERGpgc8gxdC2Ei46IiIjUwOaQY+haCBcdERERqYHNIcfQtRAuOiIiIlIDm0OOoWshXHRERESkBjaHnF2E7pkzZzBq1Ci0aNECWq0W4eHhJR6/bds2CCEQFhZm9Lnr169jxIgR8PX1haenJ/r164dLly6ZPScuOiIiIlIDm0POLkI3ISEBQUFBiI6ORsOGDUsM3du3b6NOnToICAgwGboREREICgrCpk2bkJiYiObNmyMsLAz5+flmzYmLjoiIiNTA5pCzi9AtKChQ/ndUVFSJoTtr1ix06dIFMTExRqH7zTffQAiB/fv3K7f98ssv0Gg02LRpk1lz4qIjIiIiNbA55OwidIsqKXQzMzPh7u6OH3/80WTozpo1C35+figsLDS4vVWrVoiJiTFrHlx0REREpAY2h1ylCt0+ffpgzJgxAGAydKOjo9GpUyej+w0aNAgdOnQwax5cdERERKQGNodcpQndHTt2wNfXF1euXAFgOnS7d++OPn36GN133LhxaNCgQYnPe+PGDVy8eFEZ6enpXHRERERkcQxduUoRunl5eahXrx7i4uKU22ShGxkZafSYY8eORcOGDUt83tjYWAghjAYXHREREVkSQ1euUoTu/PnzERoaiitXriAnJwc5OTkYOHAgmjdvjpycHNy9exfAo126wDO6REREZA0MXblKEboxMTEmz7bqx+rVqwHcfzFa1apVjR6zdevWfDEaERERVUhsDrlKEbqnTp1CamqqwYiIiED9+vWRmpqqvCGEfnuxlJQU5b6nT5/m9mJERERUYbE55OwidHNzcxEfH4/4+Hi0a9cOTZs2VT7Ozs42eR9T1+gC998wIjg4GJs3b8aOHTvQokULhIWFQafTmTUnLjoiIiJSA5tDzi5C9+zZs9LLElJTU03eRxa6+rcA9vHxgaenJ5577jn8/vvvZs+Ji46IiIjUwOaQs4vQrYi46IiIiEgNbA45hq6FcNERERGRGtgccgxdC+GiIyIiIjWwOeQYuhbCRUdERERqYHPIMXQthIuOiIiI1MDmkGPoWggXHREREamBzSHH0LUQLjoi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8847WrduneLj45WbmxvopYWUyy+/XLNnz9bGjRu1c+dO3X333QoPD9eqVasCvbSQ9rvf/U7nnXceoRsAc+fO1eDBg7VmzRq9++67Wr9+vZYvXx7oZYUMj8ej8PBwzZs3T3/961+1ceNGpaWlKS0tTceOHQv08oJWcXGxkpOTlZOTo7S0tC5Dd8OGDQoLC9P//d//qaysTIsXL1ZERIQqKiqcX3CIIHT7sEceeUSxsbH69ttvvfv+/Oc/q1+/fvryyy8DuLLQ8s0333Tat3DhQp1zzjkBWA2k//2kIyYmRs8++yyh67Bt27YpIiJCu3fvDvRSQtbixYuVmprq81OlsrIyWZalDz/8MIArC24nTpzw/nrGjBldhm5aWppuueUWn31XXHGFrrnmGruXF7II3T5s4sSJuvHGG332NTQ0KCwsTIWFhYFZFCRJa9askWVZ+u677wK9lJB01VVXafny5fJ4PISuw2bNmqWrr7460MsIaQsWLNCYMWN89v3973+XZVnatWtXgFYVWroK3c8//1yWZamkpMRn/5NPPqnIyEi1tLQ4uMLQQej2YQMHDtR9993Xaf+QIUN01113BWBFaDdnzhylpKQEehkhqaioSElJSWpsbCR0A+D888/Xr3/9ay1dulRxcXE666yzdN1113F+qIPef/99RURE6E9/+pMOHz6szz//XNOnT1d6erra2toCvbyQ0FXovv3227IsS3v37vXZv337dlmWpT179ji4wtBB6PZhERERysvL67T/4osv1sKFCwOwIkhSeXm5wsPDlZ+fH+ilhJzm5mYNGzZM69atkyRCNwAiIyMVGxuryy+/XG+//bZef/11XXDBBfrJT36i1tbWQC8vZLz11luKjY2VZVmyLEvjxo3TwYMHA72skNFV6L7yyiuyLKvT6W7V1dUcbbcRoduHRURE6A9/+EOn/SNHjtSiRYsCsCLU1tZqyJAhmjp1qs/5WnDGPffco/T0dO+fPaHrvP79+8vlcvn8z7z9x+abNm0K4MpCx65du5SQkKA777xTZWVlKioq0pgxY5SRkcGPxx1yutD9z3/+47O/qqqK86dtROj2YZy60Ls0NDRo1KhRGj16tA4fPhzo5YSc/fv3KzIyUm+//bYaGhrU0NCgLVu2yLIsvf/++zp69GiglxgSkpKSdNlll3XaHx8fz5VIHDJ+/Hjl5OT4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" width="639.8333333333334">

## Plotting a Bulk Magnetization visualization after adding non uniformity


```python
m2 = b.magentization(T1, T2, Bz[:5])
```


```python
m2.rotate(1)
```


```python
fig = plt.figure()
ax = fig.gca(projection='3d')

# Origin
x = np.zeros(5)
y = np.zeros(5)
z = np.zeros(5)

colors = ['r', 'b', 'y', 'g', 'c']

# Initizalizing plot
# Directions of the vector m2
u = m2.vector[0, 0]  # x Component 
v = m2.vector[0, 1]  # y Component
w = m2.vector[0, 2]   # z Component 

quiver = ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="red")

for point in range(5):
    ax.plot(m2.vector[:0, 0, point], m2.vector[:0, 1, point], m2.vector[:0, 2, point], color=colors[point], label="Trajectory %s"%(point+1))

def update(t):
    global quiver
    u = m.vector[t, 0]
    v = m.vector[t, 1]
    w = m.vector[t, 2]
    quiver.remove()
    quiver= ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1)
    for point in range(5):
        ax.plot(m2.vector[:t, 0, point], m2.vector[:t, 1, point], m2.vector[:t, 2, point],
                color=colors[point], label="Trajectory %s"%(point+1))
    
ax.set_xlim3d([-0.3, 0.3])
ax.set_xlabel('X')

ax.set_ylim3d([-0.3, 0.3])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.5, 1.5])
ax.set_zlabel('Z')

ax.view_init(elev= 28, azim=-45)
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200, blit= True)
ax.legend()
ani.save("magnetization2.gif")
plt.show()
```

![](images/bulk2.png)