class Zone:
    """
    detection zone
    """
    def __init__(self,ymin,ymax,xmin,xmax,count=0):
        self.ymin = ymin
        self.ymax = ymax
        self.xmin = xmin
        self.xmax = xmax
        self.count =  0
        self.active = False

    def count_increment(self,in_zone):
        """
        :param in_zone:
        :return:
        """
        if (not self.active) and in_zone: # zone pass from inactive to active
            self.count += 1  # counter increment
        self.active = in_zone

    def zone_in_box(self,box):
        """
        if the zone contains in a box
        :param box:
        :return:
        """
        return (box.xmin < self.xmin) and (box.ymin < self.ymin) and (box.xmax > self.xmax) \
                and (box.ymax > self.ymax)

    def center_in_zone(self,box):
            """
            if a box center is in zone
            :param box:
            :return:
            """
            box_x = (box.xmin + box.xmax)/2
            box_y = (box.ymin + box.ymax)/2
            return (box_x > self.xmin) and (box_y > self.ymin) and (box_x < self.xmax) \
               and (box_y < self.ymax)