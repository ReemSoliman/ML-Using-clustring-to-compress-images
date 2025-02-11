# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:57:55 2022

@author: reem_
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import io 
import statistics
  
def load_image(imagePath):
      
    img = io.imread(imagePath) 
    io.imshow(img)
    io.show()
   
    # scaling valus for faster processing
    img = img / 255 
  
    return img

def initialize_centroids(image, clusters):
    rows = image.shape[0]
    cols = image.shape[1]
    
    #print(rows,cols)
    
    points = image.reshape(rows*cols, 3)
    print(points)
    m,n = points.shape 
    
    centroids = np.zeros((clusters, n)) 
    print("c ", centroids)
    
   
    centroids = points[np.random.choice(range(points.shape[0]), replace = False,     size = clusters), :]
    return points, centroids
# Function to measure the mahaten
def calculate_distance(x1, y1,z1, x2, y2,z2):
      
    dist = abs(x1 - x2) + abs(y1 - y2)+ abs(z1 - z2)
    
    return dist

def k_means(points, centroids, clusters, iterations):
    
    index = np.zeros(len(points)) 
    distances = np.zeros(len(points))  #distance between centroid and each point 
    
    tempCluster = 0
    tempDistance =0
    previous_converge = 0
    converged = False 
    iteration = 0
    
    
    
    while not converged  : 
        #print("iteration", iteration)
        for p in range(0, len(points)):
           #print("point", p) 
           for c in range(clusters):
               
                   
               x1= points[p,0]
               y1 = points[p,1]
               z1 = points[p,2]
               
               x2= centroids[c,0]
               y2 = centroids[c,1]
               z2 = centroids[c,2]
               
               dist = calculate_distance(x1, y1,z1, x2, y2,z2)
               if c == 0:
                   tempDistance = dist
                   tempCluster = c
               else:
                   if(dist < tempDistance):
                       tempDistance = dist
                       tempCluster = c
               index[p] = tempCluster
               distances[p] = tempDistance
               
        for c in range(clusters):
           
            
            R=[]
            G=[]
            B=[]
            R.clear()
            G.clear()
            B.clear()
            for j in range(len(points)):
                  
                if(index[j] == c):
                    
                    R.append(points[j, 0])
                    G.append(points[j, 1])
                    B.append(points[j, 2])

            if R != [] and G != [] and B!= []:
                centroids[c, 0] = statistics.median(R) 
                centroids[c, 1] = statistics.median(G) 
                centroids[c, 2] = statistics.median(B)
            else:
                raise Exception("Empty Cluster")
         
        converge = round(sum(distances)/ len(distances),3)
        if iteration == 0:
            previous_converge = converge
            print("pre cov", previous_converge)  
            print("conv", converge)
        else:
            print("pre cov", previous_converge)  
            print("conv", converge)
            if(converge >= previous_converge) :
                converged = True
            else:
                previous_converge = converge
                converged = False
        iteration +=1
    #print("pre cov", previous_converge)  
    #print("conv", converge)
    print("# iterations ", iteration)        
    
    print("end kmeans")
    return centroids, index

def compress_image(centroids, index, img):
  
    # assign each pixel to its centroid.
    cent = np.array(centroids)
    recovered = cent[index.astype(int), :]
      
    # getting back the 3d matrix (row, col, rgb(3))
    recovered = np.reshape(recovered, (img.shape[0], img.shape[1], img.shape[2]))
  
    # plotting the compressed image.
    plt.imshow(recovered)
    plt.show()
  
    # saving the compressed image.
    io.imsave('superman_' + str(clusters) + '_colors_manhatenl1_median_converge.png', recovered)


start = time.time()
clusters = 16 
image = load_image('homework1/data/superman.bmp')
#print("read image")
success = False

while not success and clusters>0:
    try:
        points, centroids = initialize_centroids(image,clusters)
        #print("centroids after ini", centroids)
        #print("after initialize")
        centroids, index=  k_means(points, centroids, clusters, 10)
        success = True
        
    except:
        clusters -=1
        

print("final count of clusters ", clusters)


compress_image(centroids, index, image)
#print("end")
end = time.time()
print("time elapsed: ", end - start)