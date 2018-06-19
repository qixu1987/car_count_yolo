"""class of detection zone """
class Zone:
    def __init__(self,xmin,ymin,xmax,ymax,count=0):
        self.ymin = ymin
        self.ymax = ymax
        self.xmin = xmin
        self.xmax = xmax
        self.count =  count # nb of object passed through the zone
        self.active = False

    def count_increment(self,in_zone):
        """
        :param in_zone
        :return:
        """
        if (not self.active) and in_zone: # zone pass from inactive to active
            self.count += 1  # counter increment
        self.active = in_zone

    def zone_in_box(self,box):
        """
        if the zone contained in a box
        :param box
        :return:boolean
        """
        return (box.xmin < self.xmin) and (box.ymin < self.ymin) and (box.xmax > self.xmax) \
                and (box.ymax > self.ymax)

    def center_in_zone(self,box):
        """
        if a box center is in the zone
        :param box
        :return:boolean
        """
        box_x = (box.xmin + box.xmax)/2
        box_y = (box.ymin + box.ymax)/2
        return (box_x > self.xmin) and (box_y > self.ymin) and (box_x < self.xmax) \
           and (box_y < self.ymax)

    def center_in_zone_array(self,pos_array):
        """
        if a box center is in zone
        :param pos_array
        :return:boolean
        """
        x = (pos_array[0] + pos_array[2])/2
        y = (pos_array[1] + pos_array[3])/2
        return (x > self.xmin) and (y > self.ymin) and (x < self.xmax) \
               and (y < self.ymax)