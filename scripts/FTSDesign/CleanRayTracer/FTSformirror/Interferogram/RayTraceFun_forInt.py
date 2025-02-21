'''This is a cleaned up version of RayTraceFunCleaner.py. It has also been shortened to ONLY include the functions that are necessary, rather than ones that were used when building this.  Mira'''
import numpy as np
import numpy
from random import uniform
import random

'''Rotations: Give angle wanted rotated to respective function, returns rotated point(s).'''

def Rx(x):
    Rx = np.matrix([[1,0,0],[0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return Rx
def Ry(y):
    Ry = np.matrix([[np.cos(y),0,np.sin(y)],[0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return Ry
def Rz(z):
    Rz = np.matrix([[np.cos(z), - np.sin(z), 0],[np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz
def Rxyz (thet):
    Rxyz = Rx(thet[0])*Ry(thet[1])*Rz(thet[2])
    return Rxyz

#Functions for Making the Center Ellipse(s)

''' EllipseLineInt(ELI): Give point of the line, vector of the line, and coefficients of the ellipse, find the intersection(s) of the line and the ellipsoid (assuming ellipse is rotated about the x-axis. '''
def ELI(pli,v1,coeffellipse):
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[1]**2))
    B1 = -(2*pli[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[1]**2)
    B2 = (2*v1[1]*pli[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli[2])/((v1[0])*coeffellipse[1]**2)
    B = B1 + B2
    C1 = ((pli[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[1]**2)
    C2 = -(2*pli[0]*v1[1]*pli[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli[0]*v1[2]*pli[2])/((v1[0])*(coeffellipse[1]**2))
    C3 = (pli[2]**2)/(coeffellipse[1]**2) + (pli[2]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli[0])/(v1[0]))*v1[1] + pli[1],((xint[1] - pli[0])/(v1[0]))*v1[1] + pli[1]]
    zint = [((xint[0] - pli[0])/(v1[0]))*v1[2] + pli[2], ((xint[1] - pli[0])/(v1[0]))*v1[2] + pli[2]]
    return xint,yint,zint 

'''NormalP: Given a point, vector, and ellipse, finds the point of intersection and the normal of the corresponding tangent plane.'''
def NormalP(pli,v1,coeffellipse):
    xint1, yint1, zint1 = ELI1(pli,v1,coeffellipse)
    cpos = [(2*xint1[0])/(coeffellipse[0]**2),(2*yint1[0])/(coeffellipse[1]**2),(2*zint1[0])/(coeffellipse[1]**2)]
    cneg = [(2*xint1[1])/(coeffellipse[0]**2),(2*yint1[1])/(coeffellipse[1]**2),(2*zint1[1])/(coeffellipse[1]**2)]
    cpos = np.array(cpos)
    cneg = np.array(cneg)
    return cpos,cneg

''' norm(N): Given a vector, returns the normal of it''' 
def N(V):
    VectL = numpy.array(V)
    VNorm = numpy.sqrt(VectL[0]**2 + VectL[1]**2 + VectL[2]**2)
    VectLNorm = ([u/VNorm for u in VectL])
    VectLNorm = numpy.array(VectLNorm)
    return VectLNorm

'''make_line (ML): given a point, vector, and length, makes the corresponding line '''
def ML(p,v,L):
    pointL = p
    VectL = numpy.array(v)
    Lwant = int(L)
    VectLNorm = N(v)
    t = numpy.linspace(0,Lwant,50) #make related to wanted length??
    x = [pointL[0]]
    y = [pointL[1]]
    z = [pointL[2]]
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

'''setrange2d(SR2): Give radius, intersection points, and origin, only keep points within the circle '''
def SR2(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti-origin[2])**2 <= xrange**2:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

'''CreateEllipseBoundShifted(CEBS): creates an ellipse with given coefficients at origin (0,0). Returns x, positive y, negative y, and z coordinates. (z is assumed to be 0 as it is 2d)'''
def CEBS(coeffellipse,length):
    xc=np.linspace(-float(length)/2,float(length)/2,100) #centers around angle
    yc1 = np.sqrt((1-(((xc)**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #pos side of ellipse
    yc2 = -np.sqrt((1-(((xc)**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #neg side of ellipse
    zc = np.linspace(0,0,100) #assumes merely in the x,y plane and uses 100 points
    return xc,yc1,yc2,zc

'''RotateStrandBoundShiftCORRECTING (RSBSC): Give angle to be rotated about the x axis, coefficients of ellipse, length to restrict ellipse, origin of shifting, and sign(pos or neg if it is on the positive or negative side of the y axis). Creates the ellipse rotated at the specific angle around the x-axis'''
def RSBSC(theta, coeffellipse,length, sign):
    Rotated = []
    xc,yc1,yc2,zc = CEBS(coeffellipse,length)
    if sign == 'pos':
        for i in range (0,100): 
            v = [xc[i], yc1[i],zc[i]] # number of original points using POSITIVE side of ellipse
            v2 = np.array(np.dot(v,Rx(theta))) #multiplied by rotation vector 
            Rotated.append(v2[0]) #rotated vectors
    if sign == 'neg':
        for i in range (0,100):
            v = [xc[i], yc2[i], zc[i]] #number of original points on NEGATIVE side of ellipse
            v2 = np.array(np.dot(v,Rx(theta)))#multiplied by rotation vector 
            Rotated.append(v2[0]) #rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,100): #recombining into arrays of x,y,z to be plotted
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

'''CreateZBoundShiftCorrecting (CZBSC): give number of ellipses wanted, half of the angle (theta) wanted (so if you want half of the ellipsoid, choose np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign (pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated Theta about the x-axis'''
def CZBSC(a,n, coeffellipse, length, sign):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a) #range from 0 to n angles in a divisions
        x,y,z = RSBSC(theta[i], coeffellipse, length, sign) #ellipse for specific angle
        x1.extend(x) #adding a new ellipse for each angle
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1 #returns all ellipses

'''FTSCEllipsoidCorrecting(FTSEC): give  number of ellipses wanted, half of the angle covered wanted (so if you want half of the ellipsoid, choose np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign (pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated + Theta and -Theta about the x-axis to create a symmetric ellipsoid on the pos and neg side of the z plane.'''
def FTSEC (a,n, coeffellipse, length, sign):
    X,Y,Z = CZBSC(a,n, coeffellipse, length, sign) #positive side of zplane
    X1,Y1,Z1 = CZBSC(a,-n, coeffellipse, length, sign) #negative side zplane
    if sign != 'pos' and sign != 'neg':
        print ('Error')
    return X,Y,Z, X1, Y1, Z1

'''Separate: given a list of points/vectors (i.e [[x1,y1,z1],[x2,y2,z2], ...] translates into a three arrays of x, y, and z values. (this is the format for the Transform function)'''
def sep(X): 
    x,y,z = [],[],[]
    if type(X[0]) is int or type(X[0]) is float or type(X[0]) is numpy.float64:
        x = X[0]
        y = X[1]
        z = X[2]
    else:
        for i in range (0,len(X)):
            x.append(X[i][0])
            y.append(X[i][1])
            z.append(X[i][2])
    return x,y,z

'''The reverse of sep. Translates three arrays of x,y,z values back into series of [x,y,z] points/vectors. '''
def sepop(x,y,z): 
    v = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x,y,z]
    else:
        for i in range (0,len(x)):
            a = [x[i],y[i],z[i]]
            v.append(a)
    return v


''' rotate (V, thetaxyz) rotates a vector about a given angle in order of (x,y,z)''' 
def rotate(point,thetaxyz):
    x = point[0]
    y = point[1]
    z = point[2]
    v = [x,y,z]
    lenvect = (x**2 + y**2 + z**2)**.5
    V = N(v)
    V2 = np.array(np.dot(V,Rxyz(thetaxyz)))
    v2f = V2[0]*lenvect
    return v2f

'''rotate (V, thetaxyz) rotates a vector about a given angle in order of (z,y,x)'''
def rotaterev(point,thetaxyz):
    x = point[0]
    y = point[1]
    z = point[2]
    v = [x,y,z]
    lenvect = (x**2 + y**2 + z**2)**.5
    V = N(v)
    VZ = np.array(np.dot(V,Rz(thetaxyz[2])))
    VZY = np.array(np.dot(VZ,Ry(thetaxyz[1])))
    VZYX = np.array(np.dot(VZY,Rx(thetaxyz[0])))
    v2f = VZYX[0]*lenvect
    return v2f

'''given a point (or vector) and an origin (a local one in global coordinates), shifts to the Local origin in Global coordinates'''
def shift(point, origin):
    x = point[0]
    y = point[1]
    z = point[2]
    x2 = x + origin[0]
    y2 = y + origin[1]
    z2 = z + origin[2]
    v2 = [x2,y2,z2]
    return v2

'''Given a point (or three arrays of x,y,z for points), the GLOBAL coordinates are transformed to LOCAL coordinates where the LOCAL coordinates are defined in terms of the GLOBAL coordinate system through its GLOBAL origin and GLOBAL rotation. Essentially transforms point(s) from global coordinate system to given local coordinate system. the origin is the LOCAL origin in GLOBAL coordinates'''
def transformGL(x,y,z,origin, thetaxyz):
    XTR = []
    YTR = []
    ZTR = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x,y,z]
        if x ==0 and y ==0 and z == 0:
            vf = shift(v,negvect(origin))
            XTR=vf[0]
            YTR=vf[1]
            ZTR=vf[2]
        else:
            v2S = shift(v,negvect(origin))
            v2RS = rotaterev(v2S,negvect(thetaxyz))
            XTR = v2RS[0]
            YTR = v2RS[1]
            ZTR = v2RS[2]
    else:
        for i in range (0, len(x)):
            v = [x[i],y[i],z[i]]
            if x[i] == 0 and y[i] ==0 and z[i] == 0:
                vf = shift(v,negvect(origin))
                XTR.append(vf[0])
                YTR.append(vf[1])
                ZTR.append(vf[2])
            else:
                v2S = shift(v,negvect(origin))
                v2RS = rotaterev(v2S,negvect(thetaxyz))
                XTR.append(v2RS[0])
                YTR.append(v2RS[1])
                ZTR.append(v2RS[2])    
    return XTR,YTR,ZTR

'''transforms point(s) from local coordinate system to corresponding glocal coordinate system. origin is the LOCAL origin in GLOBAL coordinates'''
def transformLG(x,y,z,origin, thetaxyz):
    XTR = []
    YTR = []
    ZTR = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x,y,z]
        if x ==0 and y ==0 and z == 0:
            vf = shift(v,origin)
            XTR=vf[0]
            YTR=vf[1]
            ZTR=vf[2]
        else:
            v2R = rotate(v,thetaxyz)
            v2RS = shift(v2R,origin)
            XTR = v2RS[0]
            YTR = v2RS[1]
            ZTR = v2RS[2]
    else:
        for i in range (0, len(x)):
            v = [x[i],y[i],z[i]]
            if x[i] == 0 and y[i] ==0 and z[i] == 0:
                vf = shift(v,origin)
                XTR.append(vf[0])
                YTR.append(vf[1])
                ZTR.append(vf[2])
            else:
                v2R = rotate(v,thetaxyz)
                v2RS = shift(v2R,origin)
                XTR.append(v2RS[0])
                YTR.append(v2RS[1])
                ZTR.append(v2RS[2])    
    return XTR,YTR,ZTR

''' SetRange3Both (SR3B): simply changing requirements. same as SR3IC but only if a point is within an ellipse.
def SR3B (ranges, xinti,yinti,zinti, origin): #given range, one point, origin, if it lies in or not
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    if ( (((xinti-xc)**2/xr**2) + ((yinti-yc)**2/yr**2)) <= 1 
        and (((yinti-yc)**2/yr**2) + ((zinti-zc)**2/zr**2)) <=1
        and (((zinti-zc)**2/zr**2) + ((xinti-xc)**2/xr**2)) <=1):
        return True
    else:
        return False'''
def SR3B (ranges, xinti,yinti,zinti, origin): #given range, one point, origin, if it lies in or not
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    if ( ((((xinti-xc)**2)/xr**2) + (((yinti-yc)**2)/yr**2) + (((zinti-zc)**2)/zr**2))) <= 1:
        return True
    else:
        return False

'''Negvect: negates a vector. '''
def negvect(vect):
    if type(vect[0]) is int or type(vect[0]) is float or type(vect[0]) is numpy.float64:
        vectset = [-x for x in vect]
    else:
        vectset = [[-y for y in x] for x in vect]
    return vectset

''' Spec: give the number of rays wanted, returns specular distribution of n vectors. Adapted from Meyer's Specular notebook.'''
def spec(n):
    x,y,z = [],[],[]
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
    V = []
    for i in np.arange(n):
        v = [x[i], y[i], z[i]]
        V.append(v)  
    return V

'''ReTransform (RT): given points and vectors, transforms from one coordinate system (given by thet and origin with respect to GLOBAL coordinate system) to another coordinate system (given by thet and origin with respect to GLOBAL coordinate system) '''
def RT(sourcepoints,v1,sourcethet1, ellipseorigin1, sourcethet2, ellipseorigin2):
    if len(sourcepoints) == 0:
        return [],[]
    spx,spy,spz = sep(sourcepoints)
    vx,vy,vz = sep(v1)
    vectorigin = [0,0,0] #don't shift vectors
    #LOCAL to GLOBAL
    vGx,vGy,vGz = transformLG(vx,vy,vz,vectorigin,sourcethet1)
    spGx,spGy,spGz = transformLG(spx,spy,spz,ellipseorigin1,sourcethet1)
    #GLOBAL back to SECOND LOCAL
    vfx,vfy,vfz = transformGL(vGx,vGy,vGz,vectorigin,sourcethet2)
    spfx,spfy,spfz = transformGL(spGx,spGy,spGz,ellipseorigin2,sourcethet2)
    sp = sepop(spfx,spfy,spfz)
    v2 = sepop(vfx,vfy,vfz)
    return sp, v2

'''Select Range specifically for ellipse 7 (see page 131 sheets) and 143.'''
def SR10(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2 and yinti <0:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

def SR7(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2 and yinti >0:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

'''Creates a circular source with a given radius'''
def circularsource(r): #radius
    xpoint = []
    ypoint = []
    zpoint = []
    for x in np.linspace(-r,r,10):
        for y in np.linspace(-r,r,10):
            d = numpy.sqrt((x**2) + (y**2))
            if d <= r: 
                xpoint.append(x)
                ypoint.append(y)
                zpoint.append(int(0))
    return xpoint, ypoint, zpoint


#given sourcepoint = [x,y,z] where x etc is an array
'''FormSource: give number of rays, potential source points, the GLOBAL angle, and GLOBAL origin (that the source should be made with respect to). Returns random collection of points and vectors. '''
def FS(specnum,sourcepoint,sourcethet,origin):
    originG = [0,0,0]
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        v1 = spec(specnum)
        vx,vy,vz = sep(v1)
        v1x,v1y,v1z = transformLG(vx,vy,vz,originG,sourcethet)
        p1x,p1y,p1z = shift(sourcepoint,origin)
        sp = [p1x,p1y,p1z]
        v2 = sepop(v1x,v1y,v1z)
    else: 
        v1 = spec(specnum)
        vx,vy,vz = sep(v1)
        v1x,v1y,v1z = transformLG(vx,vy,vz,originG,sourcethet)
        v2 = sepop(v1x,v1y,v1z)
        sp = []
        for i in range (0,specnum):
            spx = random.choice(sourcepoint[0])
            spy = random.choice(sourcepoint[1])
            spz = random.choice(sourcepoint[2])
            #sp1x,sp1y,sp1z = transformLG(spx,spy,spz,origin,sourcethet)
            spT = [spx,spy,spz]
            sp.append(spT)
    return sp,v2

'''creates a list of potential source points within a certain range, tilted at a certain angle, corresponding to a specific origin.'''
def specsource(r,origin,thet):
    x,y,z=circularsource(r)
    x1,y1,z1 = transformLG(x,y,z,origin,thet)
    sourcepoint = [x1,y1,z1]
    return sourcepoint

'''from the xrange being determined in GLOBAL coordinate system, translates from Global to Local. Returns center point and the xrange'''
def xrangeGL6 (x1,y1,z1,x3,y3,z3,origin,thet):
    x,y,z = [],[],[]
    x2,y2,z2 = transformGL(x1,y1,z1,origin,thet)
    x4,y4,z4 = transformGL(x3,y3,z3,origin,thet)
    x.extend(x2),x.extend(x4),y.extend(y2),y.extend(y4),z.extend(z2),z.extend(z4)
    
    xrangeL = np.sqrt((min(x) - max(x))**2)/2
    yrangeL = np.sqrt((min(y) - max(y))**2)/2
    zrangeL = np.sqrt((min(z) - max(z))**2)/2
    
    xcenter = min(x) + xrangeL
    ycenter = min(y) + yrangeL
    zcenter = min(z) + zrangeL
    
    xcenterL = [xcenter,ycenter,zcenter]
    xrangesL = [xrangeL,yrangeL,zrangeL]
    return xcenterL,xrangesL

'''from the xrange being determined in GLOBAL coordinate system, translates from Global to Local. Returns center point and the xrange. BUT random yrange to maximize area covered. (see fixing xrangeGL6)'''
def xrangeGL7 (x1,y1,z1,x3,y3,z3,origin,thet):
    x,y,z = [],[],[]
    x2,y2,z2 = transformGL(x1,y1,z1,origin,thet)
    x4,y4,z4 = transformGL(x3,y3,z3,origin,thet)
    x.extend(x2),x.extend(x4),y.extend(y2),y.extend(y4),z.extend(z2),z.extend(z4)
    
    xrangeL = np.sqrt((min(x) - max(x))**2)/2
    yrangeL = np.sqrt((min(y) - max(y))**2)/2
    zrangeL = np.sqrt((min(z) - max(z))**2)/2
    
    xcenter = min(x) + xrangeL
    ycenter = min(y) + yrangeL
    zcenter = min(z) + zrangeL
    
    xcenterL = [xcenter,ycenter,zcenter]
    xrangesL = [xrangeL,200,zrangeL]
    return xcenterL,xrangesL

def SR103di(ranges,X,Y,Z, origin): #this corrects SR103d
    xintG = []
    yintG = []
    zintG = []
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if ( ((xinti-xc)**2)/(xr**2) + ((yinti-yc)**2)/(yr**2) <= 1 
            and ((yinti-yc)**2)/(yr**2) + ((zinti-zc)**2)/(zr**2) <=1
            and ((zinti-zc)**2)/(zr**2) + ((xinti-xc)**2)/(xr**2) <=1):
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

#given original point and vector from it, figure out properties of intersection wanted
#returns x2 greater (G) than or less (L) than x1
def finddirec(p1,v,intpos,intneg,vectpos,vectneg):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    xpos = intpos[0]
    ypos = intpos[1]
    zpos = intpos[2]
    xneg = intneg[0]
    yneg = intneg[1]
    zneg = intneg[2]
    direc = [] 
    direcpos = []
    direcneg = []
    if v1 >=0:
        direc.append('G')
    if v1 < 0:
        direc.append('L')
    if v2 >=0:
        direc.append('G')
    if v2 < 0:
        direc.append('L')
    if v3 >=0:
        direc.append('G')
    if v3 < 0:
        direc.append('L')
    if xpos>=x1:
        direcpos.append('G')
    if xpos < x1:
        direcpos.append('L')
    if ypos>=y1:
        direcpos.append('G')
    if ypos < y1:
        direcpos.append('L')
    if zpos>=z1:
        direcpos.append('G')
    if zpos < z1:
        direcpos.append('L')
    if xneg>=x1:
        direcneg.append('G')
    if xneg < x1:
        direcneg.append('L')
    if yneg>=y1:
        direcneg.append('G')
    if yneg < y1:
        direcneg.append('L')
    if zneg>=z1:
        direcneg.append('G')
    if zneg < z1:
        direcneg.append('L')
    if direc == direcpos:
        return intpos, vectpos
    else:
        if direc == direcneg:
            return intneg, vectneg

    
def ELIorganize(p,v,coeffellipse):
    return p[0], p[1],p[2],v[0],v[1],v[2],coeffellipse[0],coeffellipse[1]

def ELI1(pli,v1,coeffellipse):
    x0,y0,z0,a,b,c,d,e = ELIorganize(pli,v1,coeffellipse)
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[1]**2))
    B1 = -(2*pli[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[1]**2)
    B2 = (2*v1[1]*pli[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli[2])/((v1[0])*coeffellipse[1]**2)
    B = B1 + B2
    C1 = ((pli[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[1]**2)
    C2 = -(2*pli[0]*v1[1]*pli[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli[0]*v1[2]*pli[2])/((v1[0])*(coeffellipse[1]**2))
    C3 = (pli[2]**2)/(coeffellipse[1]**2) + (pli[2]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt((B**2) - 4*A*C))/(2*A),(-B - np.sqrt((B**2) - 4*A*C))/(2*A)]
    t = [(xint[0]-x0)/a, (xint[1]-x0)/a]
    yint = [y0 +t[0]*b, y0 + t[1]*b]
    zint = [z0 + t[0]*c, z0 + t[1]*c]
    return xint,yint,zint
    
def REPCNi(coeffellipse,pli,v):
    Npos,Nneg = NormalP(pli,v,coeffellipse) #plane coefficients
    VectLNorm = N(v) #incident unit vector
    Npos = np.array([-x for x in Npos]) 
    Nneg = np.array([-x for x in Nneg])
    vectpos = VectLNorm - 2*N(Npos)*(np.dot(VectLNorm,N(Npos)))
    vectneg = VectLNorm - 2*N(Nneg)*(np.dot(VectLNorm,N(Nneg)))
    xint,yint,zint = ELI1(pli,v,coeffellipse)
    intpos = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    intneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    GoodInt,GoodVect = finddirec(pli,v,intpos,intneg,vectpos,vectneg)
    return GoodInt, GoodVect

#uses 3d ellipses for range, ellipse origin for center of DESIRED range, 
def RSEPCNi(coeffellipse,pli,vectors, ranges, ellipseorigin):
    Vect = []
    pointints = []
    if len(pli) == 0:
        return [],[]
    if type(pli[0]) is int or type(pli[0]) is float: #assuming it is a source from one point
        for i in range (0,len(vectors)):
            Gpoint,Gvect = REPCNi(coeffellipse,pli,vectors[i])
            if SR3B(ranges, Gpoint[0],Gpoint[1],Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    else:
        for i in range (0, len(pli)):
            Vi = vectors[i]
            Pli = pli[i] #(or pli/original points of lines)
            Gpoint,Gvect = REPCNi(coeffellipse,Pli,Vi)
            if SR3B(ranges, Gpoint[0],Gpoint[1],Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    return pointints, Vect

'''Ellipse Line Intersection 2: for Ellipse 4 (for some reason it works?) '''
def ELI2(p,v,coeffellipse):
    x0,y0,z0,a,b,c,d,e = ELIorganize(p,v,coeffellipse)
    A = (e**2)/(d**2) + (b**2)/(a**2) + (c**2)/(a**2)
    B = (-2*x0*b**2)/(a**2) + (2*y0*b)/(a) + (-2*x0*c**2)/(a**2) + (2*z0*b)/(a**2)
    C = ((x0**2)*(b**2))/(a**2) + (-2*y0*b*x0)/(a) + y0**2 + ((x0**2)*(c**2))/(a**2) + (-2*z0*c*x0)/(a) + z0**2 - e**2
    xint = [(-B + np.sqrt((B**2) - 4*A*C))/(2*A),(-B - np.sqrt((B**2) - 4*A*C))/(2*A)]
    t = [(xint[0]-x0)/a, (xint[1]-x0)/a]
    yint = [y0 +t[0]*b, y0 + t[1]*b]
    zint = [z0 + t[0]*c, z0 + t[1]*c]
    return xint,yint,zint

def REPCNi2(coeffellipse,pli,v):
    Npos,Nneg = NormalP(pli,v,coeffellipse) #plane coefficients
    VectLNorm = N(v) #incident unit vector
    Npos = np.array([-x for x in Npos]) 
    Nneg = np.array([-x for x in Nneg])
    vectpos = VectLNorm - 2*N(Npos)*(np.dot(VectLNorm,N(Npos)))
    vectneg = VectLNorm - 2*N(Nneg)*(np.dot(VectLNorm,N(Nneg)))
    xint,yint,zint = ELI2(pli,v,coeffellipse)
    intpos = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    intneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    GoodInt,GoodVect = finddirec(pli,v,intpos,intneg,vectpos,vectneg)
    return GoodInt, GoodVect

#uses 3d ellipses for range, ellipse origin for center of DESIRED range, 
def RSEPCNi2(coeffellipse,pli,vectors, ranges, ellipseorigin):
    Vect = []
    pointints = []
    if len(pli) == 0:
        return [],[]
    if type(pli[0]) is int or type(pli[0]) is float: #assuming it is a source from one point
        for i in range (0,len(vectors)):
            Gpoint,Gvect = REPCNi2(coeffellipse,pli,vectors[i])
            if SR3B(ranges, Gpoint[0],Gpoint[1],Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    else:
        for i in range (0, len(pli)):
            Vi = vectors[i]
            Pli = pli[i] #(or pli/original points of lines)
            Gpoint,Gvect = REPCNi2(coeffellipse,Pli,Vi)
            if SR3B(ranges, Gpoint[0],Gpoint[1],Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    return pointints, Vect
    

    
''' PlaneLineIntersectionz(PLINT): given a plane z = a number, finds intersection points of all rays'''
def PLINTz(z,p,v):
    points = []
    for i in range (0,len(p)):
        t = (z - p[i][2])/v[i][2]
        xi = p[i][0] + t*v[i][0]
        yi = p[i][1] + t*v[i][1]
        points.append([xi,yi,z])
    return points

''' PlaneLineIntersection(PLINT): given a plane y = a number, finds intersection points of all rays'''
def PLINTy(y,p,v):
    points = []
    for i in range (0,len(p)):
        t = (y - p[i][1])/v[i][1]
        xi = p[i][0] + t*v[i][0]
        zi = p[i][2] + t*v[i][2]
        points.append([xi,y,zi])
    return points

'''select region mirror. if a point is within the ellipse of a mirror, return true.'''
def SRM(p,coeffmirr,origin):
    X= p[0]
    Z = p[2]
    if ((((X-origin[0])**2)/coeffmirr[1]**2) + ((Z-origin[2])**2)/coeffmirr[0]**2) <=1:
        return True
    return False
        
'''find intersection points of rays and the mirror.'''         
def IntM(p,v,coeffmirr,originmirr):
    hitints= []
    hitvects = []
    missints = []
    missvects = []
    intpoints = PLINTy(originmirr[1],p,v)
    for i in range (0,len(intpoints)):
        if SRM(intpoints[i],coeffmirr,originmirr) == True:
            hitints.append(intpoints[i])
            VectLNorm = N(v[i])
            PNorm = [0,-1,0] #from definition of mirror (check sign what)
            VectReflect = VectLNorm -2*N(PNorm)*(np.dot(VectLNorm,N(PNorm)))
            hitvects.append(VectReflect) #change to reflected
        else:
            missints.append(intpoints[i])
            missvects.append(v[i])
    return hitints,hitvects,missints,missvects

''' for ONE ray'''
def PLINTyS(y,p,v):
    t = (y - p[1])/v[1]
    xi = p[0] + t*v[0]
    zi = p[2] + t*v[2]
    return(xi,y,zi)

'''find intersection points of one ray and the mirror.'''         
def IntMS(p,v,coeffmirr,originmirr):
    hitints= []
    hitvects = []
    missints = []
    missvects = []
    intpoint = PLINTyS(originmirr[1],p,v)
    if SRM(intpoint,coeffmirr,originmirr) == True:
        hitints = intpoint
        VectLNorm = N(v)
        PNorm = [0,-1,0] #from definition of mirror (check sign what)
        VectReflect = VectLNorm -2*N(PNorm)*(np.dot(VectLNorm,N(PNorm)))
        hitvects = VectReflect #change to reflected
    else:
        missints = intpoint
        missvects = v
    return hitints,hitvects,missints,missvects

'''plotting the mirror '''
def mirror(origin,coeffmirr,y):
    px = []
    pz = []
    py = []
    X = np.linspace(-coeffmirr[1],coeffmirr[1],50)
    Z = np.linspace(-coeffmirr[0],coeffmirr[0],50)
    for i in range (50):
        x = X[i]
        for j in range (50):
            z = Z[j]
            if ((((x-origin[0])**2)/coeffmirr[1]**2) + ((z-origin[2])**2)/coeffmirr[0]**2) <1:
                px.append(x)
                pz.append(z)
                py.append(y)
    return px,py,pz

'''plotting the polarizers '''
def polarizer(origin,coeffmirr,y):
    px = []
    pz = []
    py = []
    X = np.linspace(-coeffmirr[1],coeffmirr[1],50)
    Z = np.linspace(-coeffmirr[0],coeffmirr[0],50)
    for i in range (50):
        x = X[i]
        for j in range (50):
            z = Z[j]
            if ((((x)**2)/coeffmirr[1]**2) + ((z)**2)/coeffmirr[0]**2) <1:
                px.append(x)
                pz.append(z)
                py.append(y)
    thet = [0,0,0]
    px,py,pz = transformLG(px,py,pz,origin,thet)
    return px,py,pz


def ELI3(pli,v1,coeffellipse):
    x0,y0,z0,vx,vy,vz,a,b = ELIorganize(pli,v1,coeffellipse)
    A = 1/(a**2) + ((vy**2)/((b**2)*(vx**2))) + ((vz**2)/((b**2)*(vx**2)))
    B1 = ((-2*x0*(vy**2))/((b**2)*(vx**2))) + ((2*vy*y0)/((b**2)*(vx)))
    B2 = ((-2*x0*(vz**2))/((b**2)*(vx**2))) + ((2*vz*z0)/((b**2)*(vx)))
    B = B1 + B2
    C1 = (((x0**2)*(vy**2))/((b**2)*(vx**2))) + ((-2*x0*vy*y0)/((b**2)*(vx))) + ((y0**2)/(b**2))
    C2 = (((x0**2)*(vz**2))/((b**2)*(vx**2))) + ((-2*x0*vz*z0)/((b**2)*(vx))) + ((z0**2)/(b**2))
    C = C1 + C2 -1.0
    xint = [(-B + np.sqrt((B**2)-4*A*C))/(2*A), (-B - np.sqrt((B**2)-4*A*C))/(2*A)]
    t= [(xint[0]-x0)/vx, (xint[1]-x0)/vx]
    yint = [y0 +t[0]*vy, y0 + t[1]*vy]
    zint = [z0 + t[0]*vz, z0 + t[1]*vz]
    return xint,yint,zint

def REPCNi3(coeffellipse,pli,v):
    Npos,Nneg = NormalP(pli,v,coeffellipse) #plane coefficients
    VectLNorm = N(v) #incident unit vector
    Npos = np.array([-x for x in Npos]) 
    Nneg = np.array([-x for x in Nneg])
    vectpos = VectLNorm - 2*N(Npos)*(np.dot(VectLNorm,N(Npos)))
    vectneg = VectLNorm - 2*N(Nneg)*(np.dot(VectLNorm,N(Nneg)))
    xint,yint,zint = ELI3(pli,v,coeffellipse)
    intpos = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    intneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    GoodInt,GoodVect = finddirec(pli,v,intpos,intneg,vectpos,vectneg)
    return GoodInt, GoodVect

''' Just gives randomized polarization'''
def InitialPolarization():
    A = random.random()
    thet = A*2*np.pi
    Eox = np.cos(thet)
    Eoy = np.sin(thet)
    return Eox,Eoy,thet

''' for ONE ray'''
def PLINTyS(y,p,v):
    t = (y - p[1])/v[1]
    xi = p[0] + t*v[0]
    zi = p[2] + t*v[2]
    return(xi,y,zi)

'''given two angles (of polarization and polarizer) returns the intensity of reflected'''
def PolarizerInteractionR(Eox,Eoy,thet_polarized,PolarizerAngle):
    I = Eox**2 + Eoy**2
    thet_p = PolarizerAngle
    thet_alpha = thet_polarized - thet_p
    return np.abs(np.cos(thet_alpha)*np.sqrt(I)) #absolute value?

'''given two angles (of polarization and polarizer) returns the intensity of transmitted'''
def PolarizerInteractionT(Eox,Eoy,thet_polarized,PolarizerAngle):
    I = Eox**2 + Eoy**2
    thet_p = PolarizerAngle
    thet_alpha = thet_polarized - thet_p
    return np.abs(np.sin(thet_alpha)*np.sqrt(I))

'''returns distance between two given points'''
def dist(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

'''Given initial ray and polarizer, return the TRANSMITTED RAY. All in Global Coordinates. 
If the ray ever misses the next stop, it returns 0 and is discarded '''
def IntPolT2(Ray,coeffpolar,originpolar,PolarizerAngle): #transmitted
    if Ray is None:
        return
    thet_polarized = Ray[0] #theta
    I = Ray[1] #intensity
    p = Ray[2] #point
    v = Ray[3] #vector
    Di = Ray[4] #distance
    Ex,Ey = np.sqrt(I)*np.cos(thet_polarized),np.sqrt(I)*np.sin(thet_polarized)
    Ray_T = [] #thet, I, intpoint, vects
    intpoint = PLINTyS(originpolar[1],p,v)
    Ray_T.append(-PolarizerAngle) #same vector of course
    I_T = PolarizerInteractionT(Ex,Ey,thet_polarized,PolarizerAngle)
    Ray_T.append(I_T)
    if SRM(intpoint,coeffpolar,originpolar) == True:
        Ray_T.append(intpoint)
        Ray_T.append(v) #just transmitted as same vector (assuming)
        Df = dist(p,intpoint)
        Ray_T.append(Di + Df)
    else:
        return
    return Ray_T

'''Given initial ray and polarizer, return the REFLECTED RAY. All in Global Coordinates. 
If the ray ever misses the next stop, it returns 0 and is discarded '''
def IntPolR2(Ray,coeffpolar,originpolar,PolarizerAngle): #reflected
    if Ray is None: #just bug check
        return
    thet_polarized = Ray[0] #theta
    I = Ray[1] #intensity
    p = Ray[2] #point
    v = Ray[3] #vector
    Di = Ray[4] #distance
    Ex,Ey = np.sqrt(I)*np.cos(thet_polarized),np.sqrt(I)*np.sin(thet_polarized)
    Ray_R = [] #thet, I, intpoint, vects
    intpoint = PLINTyS(originpolar[1],p,v)
    Ray_R.append(PolarizerAngle) #same vector of course
    I_R = PolarizerInteractionR(Ex,Ey,thet_polarized,PolarizerAngle)
    Ray_R.append(I_R)
    if SRM(intpoint,coeffpolar,originpolar) == True:
        Ray_R.append(intpoint)
        VectLNorm = N(v)
        PNorm = [0,-1,0] #from definition of mirror (check sign what)
        VectReflect = VectLNorm -2*N(PNorm)*(np.dot(VectLNorm,N(PNorm)))
        Ray_R.append(VectReflect) #change to reflected
        Df = dist(p,intpoint)
        Ray_R.append(Di + Df)
    else:
        return
    return Ray_R

#gives a ray (polarization, intensity = 1, point, vector, distnace) with a random angle from a point.
def CreateRay():
    Ex,Ey,thet1 = InitialPolarization() #picks arbitrary thet and intensity 1
    sourcepoint = [-160.375,-113,0] #global
    #angle (global)
    rand = float(random.randrange(32000,96000))
    angle = rand/1000
    v = [angle,251,0] #random angle
    Ray = [thet1,1.0,sourcepoint,v,0]
    return Ray

#extending create ray into z plane
def CreateRay3D(): 
    Ex,Ey,thet1 = InitialPolarization() #picks arbitrary thet and intensity 1
    sourcepoint = [-160.375,-113,0] #global
    rand = float(random.randrange(32000,96000))
    angle = rand/1000
    rand2 = float(random.randrange(32000,96000))
    angle2 = rand2/2000
    v = [angle,251,angle2] #random angle
    Ray = [thet1,1.0,sourcepoint,v,0]
    return Ray
    

'''Give ray and everything in global, does work in local, returns in global'''
def ReflEll(Ray,thetL,originL,coeffellipse):
    Ray_Refl = []
    originG = [0,0,0] # the global origin
    thetG = [0,0,0] # rotation with respect to itself aka 0,0,0
    sourcepoint = Ray[2] #originalpoint
    v = Ray[3] #vector
    SPLi,VPLi = RT(sourcepoint,v,thetG,originG,thetL,originL) #point and vector in local coordinates
    pointsf,vectsf = REPCNi3(coeffellipse,SPLi,VPLi)
    SPLf,VPLf = RT(pointsf,vectsf,thetL,originL,thetG,originG)
    Ray_Refl.append(Ray[0] + np.pi)
    Ray_Refl.append(Ray[1])
    Ray_Refl.append(SPLf)
    Ray_Refl.append(VPLf)
    Df = dist(SPLi,SPLf)
    Ray_Refl.append(Ray[4] + Df)
    return Ray_Refl

'''find intersection points of given ray and the mirror. (ignoring missing rays)'''         
def IntM2(Ray,coeffmirr,originmirr):
    if Ray is None:
        return
    p = Ray[2]
    v = Ray[3]
    Ray_M = []
    Ray_M.append(Ray[0])
    Ray_M.append(Ray[1])
    intpoint = PLINTyS(originmirr[1],p,v)
    if SRM(intpoint,coeffmirr,originmirr) == True:
        Ray_M.append(intpoint)
        VectLNorm = N(v)
        PNorm = [0,-1,0] #from definition of mirror (check sign what)
        VectReflect = VectLNorm -2*N(PNorm)*(np.dot(VectLNorm,N(PNorm)))
        Ray_M.append(VectReflect) #change to reflected
        Ray_M.append(Ray[4])
        return Ray_M
    else:
        return
   

