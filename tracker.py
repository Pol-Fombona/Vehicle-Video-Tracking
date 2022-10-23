# Imports
import numpy as np



"""

"""

from typing import Dict


class Tracker():

    def __init__(self, maxDissapeared=20) -> None:
        # 
        self.nextObjectID = 0
        # { objectID: Centroid(x, y) }
        self.objects = Dict()
        # { objectID: # frames dissapered }
        self.dissapered = Dict()
        self.maxDissapeared = maxDissapeared

    def register(self, centroid):
        # Register new objects
        self.objects[self.nextObjectID] = centroid
        self.dissapered[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove from the dictionaries the object ID
        del self.objects[objectID]
        del self.dissapered[objectID]

    def update(self, centroids):
        #
        if not centroids:
            #
            for objectID in self.dissapered.keys():
                self.dissapered[objectID] += 1

                #
                if self.dissapered[objectID] > self.maxDissapeared:
                    self.deregister(objectID)
            
            # 
            return self.objects
    
        inputCentroids = np.zeros((len(centroids), 2), dtype="uint8")
        
        for (i, (x, y)) in enumerate(centroids):
            #
            inputCentroids[i] = (x, y)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        


