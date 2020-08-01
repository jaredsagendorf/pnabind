# third party packages
from gridData import Grid

class Interpolator(object):
    def __init__(self, fileName):
        self.grid = Grid(fileName) # stores the grid data
            
    def __call__(self, xyz):
        return self.grid.interpolated(xyz[:,0], xyz[:,1], xyz[:,2])
