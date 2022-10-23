# Imports
import numpy as np


# Global Variables
MIN_DIST = 40


class Tracker():

    def __init__(self, maxDissapeared=20) -> None:
        # 
        self.nextObjectID = 0
        # { objectID: Centroid(x, y) }
        self.objects = dict()
        # { objectID: # frames dissapered }
        self.dissapered = dict()
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
            for objectID in list(self.dissapered.keys()):
                self.dissapered[objectID] += 1

                #
                if self.dissapered[objectID] > self.maxDissapeared:
                    self.deregister(objectID)
            
            # 
            return self.objects
    
        inputCentroids = np.zeros((len(centroids), 2))
        
        for (i, (x, y)) in enumerate(centroids):
            #
            inputCentroids[i] = (x, y)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # Otherwise, 
        else:
            #
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            Distances = np.zeros((len(objectIDs), len(inputCentroids)))

            # Compute the Euclidean distance for every Centroid
            for i, (x1, y1) in enumerate(objectCentroids):
                for j, (x2, y2) in enumerate(inputCentroids):
                    Distances[i, j] = np.sqrt(np.power((x2-x1), 2) + (np.power((y2-y1), 2)))


            rows = Distances.min(axis=1).argsort()
            cols = Distances.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                #
                if Distances[row, col] <= MIN_DIST:
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.dissapered[objectID] = 0

                    #
                    usedRows.add(row)
                    usedCols.add(col)

            unusedRows = set(range(0, Distances.shape[0])).difference(usedRows)
            unusedCols = set(range(0, Distances.shape[1])).difference(usedCols)

            # More objects the frame after than this new frame
            if Distances.shape[0] >= Distances.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.dissapered[objectID] += 1

                    if self.dissapered[objectID] > self.maxDissapeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects