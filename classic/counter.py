
class Counter():

    def __init__(self, roi_1, roi_2):
        
        self.cars_in = set()
        self.cars_out = set()

        self.roi_1 = roi_1
        self.roi_2 = roi_2

        self.num_cars_in = 0
        self.num_cars_out = 0


    def centroid_is_in(self, roi, centroid):

        xc, yc = centroid[0], centroid[1]

        if (xc < roi[1][0] and xc > roi[0][0]) and (yc < roi[2][1] and yc > roi[0][1]):
            return True
        
        return False


    def update(self, centroid, objectID):
        
        if self.centroid_is_in(self.roi_1, centroid) and (objectID not in self.cars_in):
            self.num_cars_in += 1
            self.cars_in.add(objectID)

        elif self.centroid_is_in(self.roi_2, centroid) and (objectID not in self.cars_out):
            self.num_cars_out += 1
            self.cars_out.add(objectID)


    def get_counter(self):
        return (self.num_cars_in, self.num_cars_out)


