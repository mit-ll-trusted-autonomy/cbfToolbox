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
from scipy.spatial.transform import Rotation as R

class Shape(ABC):
    """Abstract class that defines requirements for various shapes"""
    @abstractmethod
    def func(self,x,offset):
        """Function that defines the shape"""
        pass

    @abstractmethod
    def plot(self,ax,x):
        pass

class Point(Shape):
    """A single point in space. If given an offset, it similar to a sphere"""
    def __init__(self, ndim=2, marker='x'):
        self.ndim = ndim
        self.radius = 0
        self.marker = marker
    
    def func(self, x, offset=0):
        """Function that defines the shape"""
        return x.T.dot(x) - offset**2

    def plot(self,ax,x,color='red'):
        """Plot the shape"""
        if len(x) == 2:
            ax.plot(x[0], x[1],self.marker,color=color,mew=3)
        elif len(x) == 3:
            ax.plot3D([x[0]],[x[1]],[x[2]],color=color, marker=self.marker)

class Ellipsoid(Shape):
    def __init__(self, axes, rotation=None, invert=False, degrees=True):
        self.axes = axes
        if len(axes) > 3:
            # Rotation not currently defined for ndim > 3
            rotation = None
            
        if rotation is None:
            self.rot_mat = np.identity(len(axes))
        elif isinstance(rotation, (int, float)):
            # counter clockwise rotation
            theta = np.radians(rotation)
            self.rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta),  np.cos(theta)]])
        else:
            if isinstance(rotation,(list,tuple)):
                rotation = np.array(rotation)
            if rotation.size == 3:
                # Euler angles
                alpha,beta,gamma = rotation
                r = R.from_euler('zyx', [[alpha, beta, gamma]], degrees=degrees)
                self.rot_mat = np.squeeze(r.as_matrix())
            else:
                # rotation matrix given
                self.rot_mat = rotation

        self.sign = -1 if invert else 1

    def __str__(self):
        return 'Ellipsoid with axes {}'.format(self.axes)
    
    def func(self,x,buffer=0.):
        M = self.calc_M(buffer)
        return (x.T.dot(M).dot(x) - 1) * self.sign
    
    def calc_M(self,buffer):
        aug_axes = [a + buffer * self.sign for a in self.axes]
        diag = [a**-2 for a in aug_axes]
        M = np.diag(diag)
        return self.rot_mat @ M @ self.rot_mat.T
    
    def plot(self,ax,x,color='red'):
        '''Plot the ellipsoid on an axis'''
        if len(self.axes) == 2:
            self.plot2d(ax,x,color)
        elif len(self.axes) == 3:
            self.plot3d(ax,x,color)

    def plot2d(self,ax,x,color='red'):
        x_rad = self.axes[0]
        y_rad = self.axes[1]

        deg = list(range(0,360,5))
        deg.append(0)
        
        xl = [x_rad * np.cos(np.radians(d)) for d in deg]
        yl = [y_rad * np.sin(np.radians(d)) for d in deg]
        xy_arr = np.array([xl, yl])

        rot_mat = self.rot_mat
        xy_arr = rot_mat @ xy_arr

        xx = xy_arr[0,:] + x[0]
        yy = xy_arr[1,:] + x[1]
        
        plt.plot(xx, yy, "-",color=color,linewidth=3)

    def plot3d(self,ax,x,color='red'):
        '''Plot the 3d ellipsoid on an axis'''

        # Radii corresponding to the coefficients:
        rx, ry, rz = self.axes

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        xx = rx * np.outer(np.cos(u), np.sin(v)) + x[0]
        yy = ry * np.outer(np.sin(u), np.sin(v)) + x[1]
        zz = rz * np.outer(np.ones_like(u), np.cos(v)) + x[2]

        xyz = self.rot_mat @ np.stack([xx,yy,zz],axis=2).reshape(100*100,3).T
        xx = xyz[0,:].reshape((100,100))
        yy = xyz[1,:].reshape((100,100))
        zz = xyz[2,:].reshape((100,100))

        # Plot:
        ax.plot_surface(xx, yy, zz,  rstride=4, cstride=4, color=color, alpha=0.5)

class Sphere(Ellipsoid):
    """The set of points that are equal distance from the center point"""
    def __init__(self, radius,invert=False,ndim=2):
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
    """A planar region consisting of all points on one side of an infinite straight line"""
    def __init__(self, n, rotation=None, degrees=True) -> None:
        if len(n) > 3:
            # Rotation not currently defined for ndim > 3
            rotation = None
        if rotation is None:
            self.rot_mat = np.identity(len(n))
        elif isinstance(rotation, (int, float)):
            # counter clockwise rotation
            theta = np.radians(rotation)
            self.rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta),  np.cos(theta)]])
        else:
            if isinstance(rotation,(list,tuple)):
                rotation = np.array(rotation)
            if rotation.size == 3:
                # Euler angles
                alpha,beta,gamma = rotation
                r = R.from_euler('zyx', [[alpha, beta, gamma]], degrees=degrees)
                self.rot_mat = np.squeeze(r.as_matrix())
            else:
                # rotation matrix given
                self.rot_mat = rotation
        
        self.n = self.rot_mat @ (n/np.linalg.norm(n))

    def __str__(self):
        return "Half plane with normal {}".format(self.normal)

    def func(self,x,offset=0):
        n = jnp.array(self.n)
        x = jnp.array(x)
        return jnp.array(n.T.dot(x) - offset, float) # This puts safe side in direction of normal vector

    def plot(self,ax,x,color='red',d=5.):
        """Plot the halfplane"""
        if len(self.n) == 2:
            self.plot2d(ax,x,color)
        elif len(self.n) == 3:
            self.plot3d(ax,x,color)

    def plot2d(self,ax,x,color='red',dist=5):
        # ax + by + c = 0
        # norm: n = [a,b]
        # d = sqrt((x0-x1)**2 + (y0-y1)**2)

        # Find 2 points on the line in both directions from x
        x0,y0 = x
        a,b = self.n
        c = -x.dot(self.n)
        d = dist

        if a == 0:
            # Horizontal line. Easy to calculate
            x1,x2 = x0 + d, x0 - d
            y1,y2 = y0, y0
        else:
            # System of equations from distance formula and plane equation solved for y1
            y1 = (np.sqrt((-2*a**2*y0 + 2*a*b*x0 + 2*b*c)**2 - 4*(a**2 + b**2)*
                        (-a**2*d**2 + a**2*x0**2 + a**2*y0**2 + 2*a*c*x0 + c**2))
                            + 2*a**2*y0 - 2*a*b*x0 - 2*b*c)/(2*(a**2 + b**2))
            x1 = -(b*y1 + c)/a
            y2 =-(np.sqrt((-2*a**2*y0 + 2*a*b*x0 + 2*b*c)**2 - 4*(a**2 + b**2)*
                        (-a**2*d**2 + a**2*x0**2 + a**2*y0**2 + 2*a*c*x0 + c**2))
                            + 2*a**2*y0 - 2*a*b*x0 - 2*b*c)/(2*(a**2 + b**2))
            x2 = -(b*y2 + c)/a

        xx = [x1, x0, x2]
        yy = [y1, y0, y2]

        plt.plot(xx, yy, color=color,linewidth=3)

    def plot3d(self,ax,x,color='red',dist=5):
        x = np.array(x)
        d = -x.dot(self.n)
        
        if self.n[2] != 0:
            # create x,y
            xx,yy= np.meshgrid(np.arange(x[0]-dist,x[0]+dist),np.arange(x[1]-dist,x[1]+dist))
            # calculate corresponding z
            zz = (-self.n[0]*xx - self.n[1]*yy - d)/self.n[2]
        else:
            # Normal has no z component, cannot divide by 0
            # create x, z
            xx,zz= np.meshgrid(np.arange(x[0]-dist,x[0]+dist),np.arange(x[2]-dist,x[2]+dist))
            # calculate corersponding y
            yy = (-self.n[0]*xx - self.n[2]*zz - d)/self.n[1]
            
        # plot the surface
        ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)        
