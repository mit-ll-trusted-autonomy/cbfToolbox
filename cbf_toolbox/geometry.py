# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# 
# This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
# 
# © 2023 Massachusetts Institute of Technology.
# 
# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
# 
# The software/firmware is provided to you on an As-Is basis
# 
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
#
# Author: Andrew Schoer, andrew.schoer@ll.mit.edu

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class Shape(ABC):
    """Abstract class that defines requirements for various shapes"""
    @abstractmethod
    def func(self,x,offset):
        """Function that defines the shape"""
        pass

    @abstractmethod
    def plot(self):
        pass

class Point(Shape):
    """A single point in space. If given an offset, it similar to a sphere"""
    def func(self, x, offset=0):
        """Function that defines the shape"""
        return x.T.dot(x) - offset**2

    def plot(self,x,color='red'):
        """Plot the shape"""
        plt.plot(x[0], x[1],'x',color=color,mew=3)

class Ellipsoid(Shape):
    """An ellipsoid is a surface that can be obtained by an affine transformation of a sphere"""
    def __init__(self,axes,rotation=0,invert=False): # rotation in degrees
        self.axes = axes
        self.rotation = rotation # allow user to pass 3x3 rotation matrix
        self.sign = -1 if invert else 1

    def __str__(self):
        return 'Ellipsoid with axes {}'.format(self.axes)

    def func(self,x,buffer):
        M = self.calc_M(buffer)
        return (x.T.dot(M).dot(x) - 1) * self.sign

    def calc_M(self,buffer):
        """Calculates the transformation matrix from ellipsoid to unit sphere"""
        aug_axes = [a + buffer * self.sign for a in self.axes]
        theta = np.radians(self.rotation)

        csn = np.cos(theta)
        sn = np.sin(theta)

        a = (csn/aug_axes[0])**2 + (sn/aug_axes[1])**2
        b = -sn*csn*(1/aug_axes[1]**2 - 1/aug_axes[0]**2)
        c = (sn/aug_axes[0])**2 + (csn/aug_axes[1])**2

        return np.array([[a,b],[b,c]])

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self,value):
        value = np.array(value)
        if value.ndim > 1:
            raise ValueError('Axes must be a one-dimensional array')
        self._axes = value

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self,value):
        if value is not None:
            value = np.array(value)
            
        self._rotation = value

    def plot(self,x0,color='red'):
        '''Plot the ellipsoid on an axis
        NOTE: ONLY PLOTTING THE FIRST 2 DIMS ON XY PLANE
        '''
        x_rad = self.axes[0]
        y_rad = self.axes[1]

        deg = list(range(0,360,5))
        deg.append(0)
        
        xl = [x_rad * np.cos(np.radians(d)) for d in deg]
        yl = [y_rad * np.sin(np.radians(d)) for d in deg]
        xy_arr = np.array([xl, yl]).transpose()

        if self.rotation is None:
            angle = 0
        elif self.rotation.size > 1:
            angle = self.rotation[3] # rotation around the z-axis
        else:
            angle = self.rotation

        theta = np.radians(angle)
        if theta % np.pi != 0:
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array(((c, s), (-s, c)))
            xy_arr = xy_arr.dot(rot_mat)

        x = xy_arr[:,0] + x0[0]
        y = xy_arr[:,1] + x0[1]
        
        plt.plot(x, y, "-",color=color,linewidth=3)
        # ax.axis('equal')

class Sphere(Ellipsoid):
    """The set of points that are equal distance from the center point"""
    def __init__(self,radius,invert=False,ndim=2):
        self.radius = radius
        super().__init__([radius]*ndim,invert=invert)

    def __str__(self):
        return 'Sphere with radius {}'.format(self.radius)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self,value):
        if value <= 0:
            raise ValueError('Radius must be positive')
        self._radius = value

class HalfPlane(Shape):
    """A planar region consisting of all points on one side of an infinite straight line,
     and no points on the other side
     """
    def __init__(self, x, y, n=None, d=None):
        
        self.k_cbf = 1.0
        self.d = d

        minx = np.min(x)
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)

        if n is not None and d is not None:
            self.n = n/np.linalg.norm(n)
            # Sign of the line function
            fsign = lambda x,y: np.sign(np.transpose(n).dot(np.array([[x,y]]).T) + d)[0]

            # Find sign of the line function at the 4 corners of the region
            fll=fsign(minx,miny)
            flh=fsign(minx,maxy)
            fhh=fsign(maxx,maxy)
            fhl=fsign(maxx,miny)

            # Make sure line is visible
            if not (fll==flh and flh==fhh and fhh==fhl):
                # Find points of the line at the intersection with the boundaries
                xp = np.zeros(n.shape)
                yp = np.zeros(n.shape)
                p=0 # counts which intersection we are looking for (first or second)
                if fll!=fhl: # south
                    yp[p]=miny
                    # solve n[0]x+n[1]y+d=0
                    xp[p]=(-n[1]*yp[p]-d)/n[0]
                    p+=1
                if fll!=flh: # west
                    xp[p]=minx
                    yp[p]=(-n[0]*xp[p]-d)/n[1]
                    p+=1
                if fhl!=fhh: # east
                    if p<2:
                        xp[p]=maxx
                        yp[p]=(-n[0]*xp[p]-d)/n[1]
                        p+=1
                if flh!=fhh: # north
                    if p<2:
                        yp[p]=maxy
                        xp[p]=(-n[1]*yp[p]-d)/n[0]
                        p+=1
            else:
                print('Line is not visible')
            self.xp = xp
            self.yp = yp
            self.state = np.array([np.mean(self.xp), np.mean(self.yp)])

        else:
            # Define the wall from bottom left of box to top right of box for ease
            self.xp = x
            self.yp = y
            self.state = np.array([np.mean(x), np.mean(y)])
            n = np.array([-np.diff(y),np.diff(x)]).squeeze()
            self.n = n/np.linalg.norm(n)
        
    def __str__(self):
        return "Half plane with normal {}".format(self.normal)

    def func(self,x,offset):
        n = jnp.array(self.n)
        x = jnp.array(x)
        return jnp.abs(n.T.dot(x)) - offset
    
    def plot(self,x0,color='red'):
        """Plot the halfplane"""
        plt.plot(self.xp, self.yp, "-r",linewidth=3)
        plt.axis('equal')
        
        
