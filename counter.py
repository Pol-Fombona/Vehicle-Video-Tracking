# Imports


# Global Variables
roi_1 = [(385, 410), (460, 410), (380, 450), (460, 450)]
roi_2 = [(385, 410), (460, 410), (380, 450), (460, 450)]


class Counter():

    def __init__(self) -> None:
        
        self.cars_in = set()
        self.cars_out = set()

        self.num_cars_in = 0
        self.num_cars_out = 0


    def centroid_is_in(self, roi, centroid):

        xc, yc = centroid[0], centroid[1]

        if xc < roi[1][0] and xc > roi[0][0]:

            pass

    def update(self, centroid, objectID):
        
        if self.centroid_is_in(roi_1, centroid):
            self.num_cars_in += 1
            self.cars_in.add(objectID)

        elif self.centroid_is_in(roi_1, centroid):
            self.num_cars_in += 1
            self.cars_in.add(objectID)

        return self.num_cars_in, self.num_cars_out


