#these are the functions used for ray tracing programs. the fun in the title refers to functions but this is also fun. :)

import numpy as np
import numpy
from random import uniform

##BASIC SOURCE AND REFLECTION OFF OF PLANES

#shape_source given an array that is h,k,r for a circle and makes the points in a source. (at z coordinate z) not statistically accurate!!  
def shape_source(x,z):
    coeff3 = numpy.array(x)
    xpoint = []
    ypoint = []
    zpoint = []
    for x in range(coeff3[0]-coeff3[2],coeff3[0]+coeff3[2]):
        for y in range(coeff3[1]-coeff3[2],coeff3[1]+coeff3[2]):
            d = numpy.sqrt(((x-coeff3[0])**2) + ((y - coeff3[1])**2))
            if d <= coeff3[2]: 
                xpoint.append(x)
                ypoint.append(y)
                zpoint.append(z)
    return xpoint, ypoint, zpoint

#rotates around the x axis
def rotated_source (theta, axis, z):
    Rotated = []
    x,y,z=shape_source(axis,z)
    for i in range (0, len(x)):
        v = [x[i], y[i], z[i]]
        v2 = np.array(np.dot(v,Rx(theta)))
        Rotated.append(v2[0])
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,len(x)):
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

#returns list of points within a certain shape (namely xpoint and ypoint)
def list_points(x,y,z): 
    points = []
    xpoint = numpy.array(x)
    ypoint = numpy.array(y)
    zpoint = numpy.array(z)
    for i in range(0,len(xpoint)-1):
        points.append([xpoint[i],ypoint[i], zpoint[i]])
    return points

#given a vector, make unit vector
def norm(V):
    VectL = numpy.array(V)
    VNorm = numpy.sqrt(VectL[0]**2 + VectL[1]**2 + VectL[2]**2)
    VectLNorm = ([u/VNorm for u in VectL])
    VectLNorm = numpy.array(VectLNorm)
    return VectLNorm

#given a point and a unit vector, makes a line of a certain length starting at that point. 
def make_line(p,v,L):
    pointL = p
    VectL = numpy.array(v)
    Lwant = int(L)
    VectLNorm = norm(v)
    t = numpy.linspace(0,Lwant,50) #make related to wanted length??
    L = numpy.sqrt(((VectLNorm[0]*t)**2 + (VectLNorm[2]*t)**2 + (VectLNorm[2]*t)**2))
    x = []
    y = []
    z = []
    for t in range (0,Lwant):
        L = numpy.sqrt(((VectLNorm[0]*t)**2 + (VectLNorm[2]*t)**2 + (VectLNorm[2]*t)**2))
        xL = pointL[0] + t*VectLNorm[0]
        yL = pointL[1] + t*VectLNorm[1]
        zL = pointL[2] + t*VectLNorm[2]
        if L <= Lwant:
            x.append(xL)
            y.append(yL)
            z.append(zL)
    return x,y,z

#given many points, and a standard unit vector, make a source with rays of finite length
def make_source1(p,v,L):
    points = numpy.array(p)
    VectL = numpy.array(v)
    VectLNorm = norm(v)
    Lwant = int(L)
    x = [[] for i in range(0,len(points))]
    y = [[] for i in range(0,len(points))]
    z = [[] for i in range(0,len(points))]
    for i in range (0,len(points)):
        points2 = points[i]
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for t in range (0,Lwant):
            L = numpy.sqrt(((VectLNorm[0]*t)**2 + (VectLNorm[2]*t)**2 + (VectLNorm[2]*t)**2))
            xL = points2[0] + t*VectLNorm[0]
            yL = points2[1] + t*VectLNorm[1]
            zL = points2[2] + t*VectLNorm[2]
            if L <= Lwant:
                xi.append(xL)
                yi.append(yL)
                zi.append(zL)
                x.append(xi)
                y.append(yi)
                z.append(zi)
    return x,y,z

#attempt to use other function to make source
def make_source(p,V,L):
    points = p
    VectL = numpy.array(V)
    Lwant = int(L)
    x = []
    y = []
    z = []
    for i in range (0,len(points)):
        xLi,yLi,zLi = make_line(points[i],V,Lwant)
        x.append(xLi)
        y.append(yLi)
        z.append(zLi)
    return x,y,z

def spec(n):
    x,y,z=[0],[0],[0]
    for i in np.arange(n):
        theta=np.arccos(uniform(-1,1))
        phi=np.random.uniform(0,2*np.pi)
        xt=np.sin(theta)*np.cos(phi)
        yt=np.sin(theta)*np.sin(phi)
        zt=np.cos(theta)
        if zt<0.:
            zt=-zt
        a=uniform(0,1)
        while a>zt:
            theta=np.arccos(uniform(-1,1))
            phi=np.random.uniform(0,2*np.pi)
            xt=np.sin(theta)*np.cos(phi)
            yt=np.sin(theta)*np.sin(phi)
            zt=np.cos(theta)
            if zt<0.:
                zt=-zt
            a=uniform(0,1)
        x=np.append(x,xt)
        y=np.append(y,yt)
        z=np.append(z,zt)
    return x,y,z
#these are the VECTORS
  
def make_specsource(p,L, theta):
    Lwant = int(L)
    x = []
    y = []
    z = []
    xspec,yspec,zspec = spec(len(p))
    for i in range (1,len(p)):
        Vi = [xspec[i], yspec[i], zspec[i]]
        Vi2 = np.array(np.dot(Vi,Rx(theta)))
        xLi,yLi,zLi = make_line(p[i],Vi2[0],Lwant)
        x.append(xLi)
        y.append(yLi)
        z.append(zLi)
    return x,y,z
#making a plane/reflecting surface. for the moment a flat surface.
#given coefficients (a,b,c,d) and the range wanted, returns plane centered at origin.
def make_plane(C,r):
    coeff1 = numpy.array(C)
    r = int(r)
    xp, yp = numpy.meshgrid(range(-r,r), range(-r,r))
    zp = (-coeff1[0]*xp - coeff1[1]*yp + coeff1[3])/coeff1[2]
    return xp,yp,zp
    
#given coefficients and range, returns the normal line and the gradient of the plane
def plane_info(C,r):
    coeff1 = numpy.array(C)
    r = int(r)
    xpl,ypl,zpl = numpy.meshgrid(range(r), range(r), range(r))
    U = coeff1[0]*xpl + coeff1[1]*ypl + coeff1[2]*zpl
    dU = coeff1[:-1] #gradient (for plane)
    #normalized normal line
    N = norm(dU)
    return dU, N

#given the coefficients of the plane and the vector and initial points for a line, will return points of intersection
def intersec_point(c,v,p):
    coeff1 = numpy.array(c)
    VectL = numpy.array(v)
    VectLNorm = norm(v)
    points = p
    #intersect = [[] for i in range(1,len(points))]
    intersect = []
    pointint = []
    for i in range (0,len(points)):
        points2 = points[i] #x_0,y_0,z_0 ORIGINAL POINT FOR INCIDENT LINES
        Cb = (coeff1[1]*VectLNorm[1])/VectLNorm[0]
        Cc = (coeff1[2]*VectLNorm[2])/VectLNorm[0]
        denom = coeff1[0] + Cb + Cc
        xint = (coeff1[3] - points2[2] - points2[1] + Cb*points2[0] + Cc*points2[0])/denom
        yint = ((xint - points2[0])/(VectLNorm[0]))*VectLNorm[1] + points2[1]
        zint = ((xint - points2[0])/(VectLNorm[0]))*VectLNorm[2] + points2[2]
        pointint.append(xint) #where line intersects
        pointint.append(yint)
        pointint.append(zint)
        intersect.append(pointint) 
    return intersect #an array of intersection points

#given the coefficients of a plane, range the initial vector and initial points for a line, will lead to properly reflected ray
def reflectneg(c,r,v,p,L):
        coeff1 = c #plane coefficients
        VectL = v #incident vector#defining points of incident vector
        VectLinit = [-a for a in v]
        VectLNorm = norm(v) #incident unit vector
        dU, N = plane_info(c,r) #gradient and normal of plane
        #reflected ray
        #VectL2 = VectLNorm + 2*N #reflected vector
        VectL2 = VectLNorm - 2*N #IS IT PLUS OR MINUS
        VectLNorm2 = norm(VectL2) #reflected unit vector
        xp,yp,zp = make_plane(c,r) #plane
        p = [p]
        pointint = intersec_point(c,v,p) #array and points of intersection
        xi,yi,zi = make_line(pointint[0],VectLinit,L) #incident line from intersection point
        xr,yr,zr = make_line(pointint[0],VectL2,L)
        return xi,yi,zi,xr,yr,zr,xp,yp,zp
    
def reflect(c,r,v,p,L):
        coeff1 = c #plane coefficients
        VectL = v #incident vector#defining points of incident vector
        #VectLinit = [-a for a in v]
        VectLNorm = norm(v) #incident unit vector
        dU, N = plane_info(c,r) #gradient and normal of plane
        #reflected ray
        VectL2 = VectLNorm + 2*N #reflected vector
        #VectL2 = VectLNorm - 2*N #IS IT PLUS OR MINUS
        VectLNorm2 = norm(VectL2) #reflected unit vector
        xp,yp,zp = make_plane(c,r) #plane
        p = [p]
        pointint = intersec_point(c,v,p) #array and points of intersection
        xi,yi,zi = make_line(pointint[0],v,L) #incident line from intersection point
        xr,yr,zr = make_line(pointint[0],VectL2,L)
        return xi,yi,zi,xr,yr,zr,xp,yp,zp

#given the coefficients of a plane, range, the initial vectorS and initial pointS for lineS, will lead to properly reflected rayS
def reflect_source(c,r,v,p,L):
    points = p
    xi = []
    yi = []
    zi = []
    xr = []
    yr = []
    zr = []
    xp = []
    yp = []
    zp = []
    for i in range (0,len(points)):
        xiL,yiL,ziL,xrL,yrL,zrL,xpL,ypL,zpL = reflect(c,r,v,p[i],L)
        xi.append(xiL)
        yi.append(yiL)
        zi.append(ziL)
        xr.append(xrL)
        yr.append(yrL)
        zr.append(zrL)
        xp.append(xpL)
        yp.append(ypL)
        zp.append(zpL)
    return xi,yi,zi,xr,yr,zr,xp,yp,zp

##ROTATION MATRICES
#Rotation matrices around a given axis (in this case x,y,z)
#variable given is the angle of rotation wanted 
def Rx(x):
    Rx = np.matrix([[1,0,0],[0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return Rx

def Ry(y):
    Ry = np.matrix([[np.cos(y),0,np.sin(y)],[0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return Ry

def Rz(z):
    Rz = np.matrix([[np.cos(z), - np.sin(z), 0],[np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz


##CREATING ELLIPSOIDS
#create an ellipse (centered at (0,0,0))
def createellipse(coeffellipse):
    xc=np.linspace(-coeffellipse[0],coeffellipse[0],100)
    yc1 = np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    yc2 = -np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    zc = np.linspace(0,0,100)
    return xc,yc1,yc2,zc

#create a RESTRICTED ellipse (centered at (0,0,0)) (bounded in x)
def createellipsebounded(coeffellipse,length):
    xc=np.linspace(-float(length)/2,float(length)/2,100)
    yc1 = np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    yc2 = -np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    zc = np.linspace(0,0,100)
    return xc,yc1,yc2,zc

#create a RESTRICTED ellipse (centered at (0,0,0))
def createellipseboundshift(coeffellipse,length, origin):
    xc=np.linspace(origin[0]-float(length)/2,origin[0]+float(length)/2,100)
    yc1 = np.sqrt((1-(((xc-origin[0])**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    yc2 = -np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    zc = np.linspace(0,0,100)
    return xc,yc1,yc2,zc

#rotates a given ellipse around the x axis
def rotatestrand(theta, coeffellipse):
    Rotated = []
    xc,yc1,yc2,zc = createellipse(coeffellipse)
    for i in range (0,100): 
        v = [xc[i], yc1[i],zc[i]] # number of original points
        v2 = np.array(np.dot(v,Rx(theta))) #multiplied by rotation vector
        Rotated.append(v2[0]) #rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,100):
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

#rotates a given ellipse around the x axis
def rotatestrandbounded(theta, coeffellipse,length):
    Rotated = []
    xc,yc1,yc2,zc = createellipsebounded(coeffellipse,length)
    for i in range (0,100): 
        v = [xc[i], yc1[i],zc[i]] # number of original points
        v2 = np.array(np.dot(v,Rx(theta))) #multiplied by rotation vector
        Rotated.append(v2[0]) #rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,100):
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

#makes the ellipse 3d by rotating around the x axis
def createz(a,n, coeffellipse):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a)
        x,y,z = rotatestrand(theta[i], coeffellipse)
        x1.extend(x)
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1

#'negative' ellipse, by rotating it on the other side of the x-axis
def createzneg(a,n, coeffellipse):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a)
        x,y,z = rotatestrand(np.pi + theta[i], coeffellipse)
        x1.extend(x)
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1

#creates ellipse only within x-range
def createzbounded(a,n, coeffellipse, length):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a)
        x,y,z = rotatestrandbounded(theta[i], coeffellipse, length)
        x1.extend(x)
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1

#reverse/negative bound ellipsoids
def createzboundedneg(a,n, coeffellipse, length):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a)
        x,y,z = rotatestrandbounded(np.pi + theta[i], coeffellipse, length)
        x1.extend(x)
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1

#combination of all of the above functions. Given angle, coefficients, x-range (length of x), origin, and if it should be a regular or reversed ellipsoid, creates an ellipsoid that is symmetric about the x-axis
def FTSCEllipsoid (a,n, coeffellipse, length, origin, sign):
    if sign == 'pos':
        X,Y,Z = createzboundshift(a,n, coeffellipse, length, origin)
        X1,Y1,Z1 = createzboundshift(a,-n, coeffellipse, length, origin)
    if sign == 'neg':
        X,Y,Z = createzboundshiftneg(a,n, coeffellipse, length, origin)
        X1,Y1,Z1 = createzboundshiftneg(a,-n, coeffellipse, length, origin)
    if sign != 'pos' and sign != 'neg':
        print ('Error')
    return X,Y,Z, X1, Y1, Z1


##ELLIPSE AND RAY INTERSECTION

#given point of line, vector, and the axes of the ellipse, find the intersection(s)
def ellipselineint(pli,v1,coeffellipse): #given point of line, vector, and the axes of the ellipse, find the intersection
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[0]**2))
    B1 = -(2*pli[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    B2 = (2*v1[1]*pli[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli[2])/((v1[0])*coeffellipse[0]**2)
    B = B1 + B2
    C1 = ((pli[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    C2 = -(2*pli[0]*v1[1]*pli[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli[0]*v1[2]*pli[2])/((v1[0])*(coeffellipse[0]**2))
    C3 = (pli[2]**2)/(coeffellipse[0]**2) + (pli[1]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli[0])/(v1[0]))*v1[1] + pli[1],((xint[1] - pli[0])/(v1[0]))*v1[1] + pli[1]]
    zint = [((xint[0] - pli[0])/(v1[0]))*v1[2] + pli[2], ((xint[1] - pli[0])/(v1[0]))*v1[2] + pli[2]]
    return xint,yint,zint

#gets the coefficients of the tangent plane
def tangcoef(pli,v1,coeffellipse):
    xint1, yint1, zint1 = ellipselineint(pli,v1,coeffellipse)
    c = [xint1[0]/(coeffellipse[0]**2),yint1[0]/(coeffellipse[1]**2),zint1[0]/(coeffellipse[0]**2),1]
    return c

#makes the tangent plane at an intersection point given the point, vector,  ellipse coefficients, and size
def tangplane(pli,v1,coeffellipse, r):
    c = tangcoef(pli,v1,coeffellipse)
    r = int(r)
    xp,yp,zp = make_plane(c, r)
    return xp,yp,zp

#given the coefficients of a plane, range the initial vector and initial points for a line, will lead to properly reflected ray
def reflectellipse(coeffellipse,r,v,p,L):
        c = tangcoef(p,v,coeffellipse) #plane coefficients
        VectL = v #incident vector#defining points of incident vector
        VectLinit = [-a for a in v]
        VectLNorm = norm(v) #incident unit vector
        dU, N = plane_info(c,r) #gradient and normal of plane
        #reflected ray
        #VectL2 = VectLNorm + 2*N #reflected vector
        VectL2 = VectLNorm - 2*N #IS IT PLUS OR MINUS
        VectLNorm2 = norm(VectL2) #reflected unit vector
        xp,yp,zp = make_plane(c,r) #plane
        xint,yint,zint = ellipselineint(p,v,coeffellipse)
        pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
        xi,yi,zi = make_line(pointint,VectLinit,L) #incident line from intersection point
        xr,yr,zr = make_line(pointint,VectL2,L)
        return xi,yi,zi,xr,yr,zr,xp,yp,zp
    
#source in ellipse!
def reflect_sourceellipse(coeffellipse,r,v,p,L):
    points = p
    xi = []
    yi = []
    zi = []
    xr = []
    yr = []
    zr = []
    xp = []
    yp = []
    zp = []
    for i in range (0,len(points)):
        xiL,yiL,ziL,xrL,yrL,zrL,xpL,ypL,zpL = reflectellipse(coeffellipse,r,v,p[i],L)
        xi.append(xiL)
        yi.append(yiL)
        zi.append(ziL)
        xr.append(xrL)
        yr.append(yrL)
        zr.append(zrL)
        xp.append(xpL)
        yp.append(ypL)
        zp.append(zpL)
    return xi,yi,zi,xr,yr,zr,xp,yp,zp

#create a RESTRICTED ellipse BUT CANTERED AT THE ORIGIN (pg 92)
def createellipseboundshift(coeffellipse,length, origin):
    xc=np.linspace(origin[0]-float(length)/2,origin[0]+float(length)/2,100)
    yc1 = np.sqrt((1-(((xc-origin[0])**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) + origin[1]
    yc2 = -np.sqrt((1-(((xc-origin[0])**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) + origin[1]
    zc = np.linspace(0,0,100)
    return xc,yc1,yc2,zc


#rotates a given ellipse around the x axis
def rotatestrandboundshift(theta, coeffellipse,length,origin):
    Rotated = []
    xc,yc1,yc2,zc = createellipseboundshift(coeffellipse,length,origin)
    for i in range (0,100): 
        v = [xc[i], yc1[i],zc[i]] # number of original points
        v2 = np.array(np.dot(v,Rx(theta))) #multiplied by rotation vector
        Rotated.append(v2[0]) #rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,100):
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

def createzboundshift(a,n, coeffellipse, length, origin):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a)
        x,y,z = rotatestrandboundshift(theta[i], coeffellipse, length, origin)
        x1.extend(x)
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1

def createzboundshiftneg(a,n, coeffellipse, length, origin):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a)
        x,y,z = rotatestrandboundshift(np.pi + theta[i], coeffellipse, length, origin)
        x1.extend(x)
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1


def ellipselineintshift(pli,v1,coeffellipse,origin): #given point of line, vector, and the axes of the ellipse, find the intersection
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[0]**2))
    B1 = -(2*pli[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    B2 = (2*v1[1]*pli[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli[2])/((v1[0])*coeffellipse[0]**2)
    B = B1 + B2
    C1 = ((pli[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    C2 = -(2*pli[0]*v1[1]*pli[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli[0]*v1[2]*pli[2])/((v1[0])*(coeffellipse[0]**2))
    C3 = (pli[2]**2)/(coeffellipse[0]**2) + (pli[1]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli[0])/(v1[0]))*v1[1] + pli[1],((xint[1] - pli[0])/(v1[0]))*v1[1] + pli[1]]
    zint = [((xint[0] - pli[0])/(v1[0]))*v1[2] + pli[2], ((xint[1] - pli[0])/(v1[0]))*v1[2] + pli[2]]
    xintshift = [x + origin[0] for x in xint]
    yintshift = [y + origin[1] for y in yint]
    zintshift = zint
    return xintshift,yintshift,zintshift

def tangcoefshift(pli,v1,coeffellipse, origin):
    xint1, yint1, zint1 = ellipselineintshift(pli,v1,coeffellipse, origin)
    c = [xint1[0]/(coeffellipse[0]**2),yint1[0]/(coeffellipse[1]**2),zint1[0]/(coeffellipse[0]**2),1]
    return c

def reflectellipseshift(coeffellipse,r,v,p,L, origin):
        c = tangcoefshift(p,v,coeffellipse, origin) #plane coefficients
        VectL = v #incident vector#defining points of incident vector
        VectLinit = [-a for a in v]
        VectLNorm = norm(v) #incident unit vector
        dU, N = plane_info(c,r) #gradient and normal of plane
        #reflected ray
        #VectL2 = VectLNorm + 2*N #reflected vector
        VectL2 = VectLNorm - 2*N #IS IT PLUS OR MINUS
        VectLNorm2 = norm(VectL2) #reflected unit vector
        xp,yp,zp = make_plane(c,r) #plane
        xint,yint,zint = ellipselineintshift(p,v,coeffellipse, origin)
        pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
        xi,yi,zi = make_line(pointint,VectLinit,L) #incident line from intersection point
        xr,yr,zr = make_line(pointint,VectL2,L)
        return xi,yi,zi,xr,yr,zr,xp,yp,zp

def reflect_sourceellipseshift(coeffellipse,r,v,p,L, origin):
    points = p
    xi = []
    yi = []
    zi = []
    xr = []
    yr = []
    zr = []
    xp = []
    yp = []
    zp = []
    for i in range (0,len(points)):
        xiL,yiL,ziL,xrL,yrL,zrL,xpL,ypL,zpL = reflectellipseshift(coeffellipse,r,v,p[i],L, origin)
        xi.append(xiL)
        yi.append(yiL)
        zi.append(ziL)
        xr.append(xrL)
        yr.append(yrL)
        zr.append(zrL)
        xp.append(xpL)
        yp.append(ypL)
        zp.append(zpL)
    return xi,yi,zi,xr,yr,zr,xp,yp,zp
