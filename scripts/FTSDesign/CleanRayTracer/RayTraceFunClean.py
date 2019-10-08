#This is a cleaned up version of RayTraceFun.py. Mira
import numpy as np
import numpy
from random import uniform

#Rotations: Give angle wanted rotated to respective function, returns rotated point(s).

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

'''Important note: Coefficients of Ellipse refers to [a,b,c] with regards to the triangle between the ellipse, center, and focus, where c is distance from focus to origin, b is furthest distance from edge to origin at right angle, and a is hypotenuse of resulting triangle''' 

'''CreateEllipseBoundShifted (CEBS): give coefficients of ellipse, the length (on x axis) wanted, and the origin this should be centered on. Creates restricted (bound and shifted) ellipse.'''
def CEBS(coeffellipse,length, origin):
    xc=np.linspace(origin[0]-float(length)/2,origin[0]+float(length)/2,100) #centers around angle
    yc1 = np.sqrt((1-(((xc-origin[0])**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #pos side of ellipse
    yc2 = -np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #neg side of ellipse
    zc = np.linspace(0,0,100) #assumes merely in the x,y plane and uses 100 points
    return xc,yc1,yc2,zc

'''RotateStrandBoundShiftCORRECTING (RSBSC): Give angle to be rotated about the x axis, coefficients of ellipse, length to restrict ellipse, origin of shifting, and sign(pos or neg if it is on the positive or negative side of the y axis). Creates the ellipse rotated at the specific angle around the x-axis'''
def RSBSC(theta, coeffellipse,length,origin, sign):
    Rotated = []
    xc,yc1,yc2,zc = CEBS(coeffellipse,length,origin)
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
def CZBSC(a,n, coeffellipse, length, origin, sign):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a) #range from 0 to n angles in a divisions
        x,y,z = RSBSC(theta[i], coeffellipse, length, origin, sign) #ellipse for specific angle
        x1.extend(x) #adding a new ellipse for each angle
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1 #returns all ellipses

'''FTSCEllipsoidCorrecting(FTSEC): give  number of ellipses wanted, half of the angle covered wanted (so if you want half of the ellipsoid, choose np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign (pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated + Theta and -Theta about the x-axis to create a symmetric ellipsoid on the pos and neg side of the z plane.'''
def FTSEC (a,n, coeffellipse, length, origin, sign):
    X,Y,Z = CZBSC(a,n, coeffellipse, length, origin, sign) #negative side of zplane
    X1,Y1,Z1 = CZBSC(a,-n, coeffellipse, length, origin, sign) #negative side zplane
    if sign != 'pos' and sign != 'neg':
        print ('Error')
    return X,Y,Z, X1, Y1, Z1

'''setrange2d(SR2): Give radius, intersection points, and origin, only keep points within the circle '''
def SR2(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

#To make reflections off center ellipse for ORIGINAL SPECULAR SOURCE (from a point)

''' spec: give the number of rays wanted, returns specular distribution of n vectors. Adapted from Meyer's Specular notebook. '''
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

''' EllipseLineInt(ELI): Give point of the line, vector of the line, and coefficients of the ellipse, find the intersection(s) of the line and the ellipsoid (assuming ellipse is rotated about the x-axis. '''
def ELI(pli,v1,coeffellipse):
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

'''TangCoef (TC): give point of intersection of an ellipse, returns the coefficients of the tangent plane (for BOTH intersection points) '''
def TC(pli,v1,coeffellipse):
    xint1, yint1, zint1 = ELI(pli,v1,coeffellipse)
    cpos = [xint1[0]/(coeffellipse[0]**2),yint1[0]/(coeffellipse[1]**2),zint1[0]/(coeffellipse[0]**2),1]
    cneg = [xint1[1]/(coeffellipse[0]**2),yint1[1]/(coeffellipse[1]**2),zint1[1]/(coeffellipse[0]**2),1]
    c = [cpos, cneg]
    return c

''' norm(N): Given a vector, return the normal of it''' 
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

''' plane_info (PI): Given the coefficients of a plane, get the normal. '''
def PI(C):
    coeff1 = numpy.array(C)
    r = int(100)
    xpl,ypl,zpl = numpy.meshgrid(range(r), range(r), range(r))
    U = coeff1[0]*xpl + coeff1[1]*ypl + coeff1[2]*zpl
    dU = coeff1[:-1] #gradient (for plane)
    #normalized normal line
    Nor = N(dU)
    return dU, Nor

'''make_plane(MP): given coefficients of plane, returns the plane. to be used for a surface plot.'''
def MP(C):
    coeff1 = numpy.array(C)
    r = int(100)
    xp, yp = numpy.meshgrid(range(-r,r), range(-r,r))
    zp = (-coeff1[0]*xp - coeff1[1]*yp + coeff1[3])/coeff1[2]
    return xp,yp,zp

'''ReflectEllipsePoint (REP): Give coefficients of ellipse, vector of the initial ray and the source of the initial ray. Returns the intial ray, reflected ray, plane coefficients, and intersections for both positive and negative. '''
def REP(coeffellipse,v,p): #p = origin is origin of source! (source origin)
    c = TC(p,v,coeffellipse) #plane coefficients
    cpos = c[0] #coefficients of positive intersection
    cneg = c[1] #coefficients of negative intersection
    VectLinit = v #initial vector
    VectLNorm = N(v) #incident unit vector
    dU, Npos = PI(cpos) #gradient and normal of plane of positive intersection
    dUneg, Nneg = PI(cneg) #gradient and normal of plane of negative intersection
    VectL2 = VectLNorm - 2*Npos #reflected vector of positive intersection
    #VectLNorm2 = N(VectL2) #reflected unit vector of positive intersection
    VectL2neg = VectLNorm - 2*Nneg
    #VectLNorm2neg = N(VectL2neg)
    #xp,yp,zp = MP(cpos) #plane of positive intersection
    #xp,yp,zp = MP(cneg) #plane of negative intersection
    xint,yint,zint = ELI(p,v,coeffellipse)
    pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    pointintneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    L1 = np.sqrt((pointint[0] - p[0])**2 + (pointint[1] - p[1])**2 + (pointint[2] - p[2])**2)
    L1neg = np.sqrt((pointintneg[0] - p[0])**2 + (pointintneg[1] - p[1])**2 + (pointintneg[2] - p[2])**2)
    xi,yi,zi = ML(p,VectLinit,L1) #incident line from positive intersection point
    xr,yr,zr = ML(pointint,VectL2,L1) #reflective line from positive intersection point
    xin,yin,zin = ML(p,VectLinit,L1neg) #incident line from negative intersection point
    xrn,yrn,zrn = ML(pointintneg,VectL2neg,L1neg) #reflective line from negative intersection point
    return xi,yi,zi,xr,yr,zr,xin,yin,zin,xrn,yrn,zrn,pointint,pointintneg

'''SetRange3dIndividual (SR3I): give a radius and an origin and if a point is within that circle and on the correct side of the x axis (pos or neg) return true. '''
def SR3I (xrange, X,Y,Z, origin, sign):
    if sign == 'pos':
        if (X-origin[0])**2 + (Z-origin[2])**2 < xrange**2 and Y > 0:
            return True
        return False
    if sign == 'neg':
        if (X-origin[0])**2 + (Z-origin[2])**2 < xrange**2 and Y < 0:
            return True
        return False
    if sign != 'pos' and sign != 'neg':
        print ('Error')

    '''ReflectSpecOriginEllipsePoint(RSOEP): give ellipse coefficients, origin of source,angle of source (usually pi/2 if going into the positive ellipse), number of rays wanted, radius of target (see SR3I), center of target (see SR3I) and if it is on the negative or positive side of the x axis. Returns all incident rays, reflecting rays, and intersection points that fit the set criteria. '''
def RSOEP(coeffellipse,sourceorigin,spectrumtheta, specnum, xrange,rangecenter, sign):
    xi = []
    yi = []
    zi = []
    xr = []
    yr = []
    zr = []
    pointints = []
    Vect = []
    xspec,yspec,zspec = spec(specnum)
    for i in range (1, len(xspec)): #does for every single ray hence time consuming
        Vi = [xspec[i], yspec[i], zspec[i]]
        Vi2 = np.array(np.dot(Vi,Rx(spectrumtheta)))
        xiL,yiL,ziL,xrL,yrL,zrL,xinL,yinL,zinL,xrnL,yrnL,zrnL, pointint, pointintneg = REP(coeffellipse,Vi2[0],sourceorigin)
        if SR3I(xrange, pointint[0], pointint[1], pointint[2], rangecenter, sign) == True: 
            xi.append(xiL)
            yi.append(yiL)
            zi.append(ziL)
            xr.append(xrL)
            yr.append(yrL)
            zr.append(zrL)
            pointints.append(pointint)
            Vect.append(Vi2[0])
        if SR3I(xrange, pointintneg[0], pointintneg[1], pointintneg[2], rangecenter, sign) == True:
            xi.append(xinL)
            yi.append(yinL)
            zi.append(zinL)
            xr.append(xrnL)
            yr.append(yrnL)
            zr.append(zrnL)
            pointints.append(pointintneg)
            Vect.append(Vi2[0])
    return xi,yi,zi,xr,yr,zr, pointints, Vect

#To make reflections AFTER the original source in the center ellipse(s). Essentially modifications of previous functions. 

'''EllipseLineIntShiftCorrecting(ELISC): give point on line, vector of line, coefficients of ellipse (assuming rotated around x axis), and shifted origin (assuming origin is still at z = 0). Returns intersection points. '''
def ELISC(pli,v1,coeffellipse,origin): #given point of line, vector, and the axes of the ellipse, find the intersection
    pli1 = [pli[0] - origin[0], pli[1] - origin[1], pli[2]] #shifting to (0,0,0) with respect to ellipse.
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[0]**2))
    B1 = -(2*pli1[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli1[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    B2 = (2*v1[1]*pli1[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli1[2])/((v1[0])*coeffellipse[0]**2)
    B = B1 + B2
    C1 = ((pli1[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli1[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    C2 = -(2*pli1[0]*v1[1]*pli1[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli1[0]*v1[2]*pli1[2])/((v1[0])*(coeffellipse[0]**2))
    C3 = (pli1[2]**2)/(coeffellipse[0]**2) + (pli1[1]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli1[0])/(v1[0]))*v1[1] + pli1[1],((xint[1] - pli1[0])/(v1[0]))*v1[1] + pli1[1]]
    zint = [((xint[0] - pli1[0])/(v1[0]))*v1[2] + pli1[2], ((xint[1] - pli1[0])/(v1[0]))*v1[2] + pli1[2]]
    xintshift = [x + origin[0] for x in xint]
    #yintshift = [y + origin[1] for y in yint]
    yintshift = yint #does y not change what?
    zintshift = zint
    #zintshift = [z + origin[2] for z in zint] (typically centered at zero)
    return xintshift,yintshift,zintshift

'''ReflectEllipsePointCorrecting(REPC): give ellipse coefficients, point on line, vector of line, ellipse origin, and sign. Returns incident rays, reflecting rays, and intersection points for positive and negative intersection. Uses sign to determine correct direction of reflecting rays.   ''' 
def REPC(coeffellipse,pli,v, ellipseorigin, sign):
    c = TC(pli,v,coeffellipse) #plane coefficients
    cpos = c[0]
    cneg = c[1]
    VectL = v
    VectLNorm = N(v) #incident unit vector
    dU, Npos = PI(cpos) #gradient and normal of plane of positive int
    dUneg, Nneg = PI(cneg)
    VectL2 = VectLNorm - 2*Npos
    VectL2neg = VectLNorm-2*Nneg
    #check if it is going in the right direction for positive int
    if sign == 'neg' and VectL2[1] < 0 : 
        VectL2 = [-x for x in VectL2]
    if sign == 'pos' and VectL2[1] > 0:
        VectL2 = [-x for x in VectL2]
    #check if it is going in the right direction for negative int
    if sign == 'neg' and VectL2neg[1] < 0 : 
        VectL2neg = [-x for x in VectL2neg]
    if sign == 'pos' and VectL2neg[1] > 0:
        VectL2neg = [-x for x in VectL2neg]
    #VectLNorm2 = norm(VectL2) #reflected unit vector
    #xp,yp,zp = MP(c,r) #plane
    xint,yint,zint = ELISC(pli,v,coeffellipse, ellipseorigin)
    #pointintRETURN = [xint,yint,zint]
    pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    pointintneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    L1 = np.sqrt((pointint[0] - pli[0])**2 + (pointint[1] - pli[1])**2 + (pointint[2] - pli[2])**2)
    L1neg = np.sqrt((pointintneg[0] - pli[0])**2 + (pointintneg[1] - pli[1])**2 + (pointintneg[2] - pli[2])**2)
    xi,yi,zi = ML(pli,VectL,L1) #incident line from intersection point
    xr,yr,zr = ML(pointint,VectL2,L1)
    xin,yin,zin = ML(pli,VectL,L1neg) #incident line from intersection point
    xrn,yrn,zrn = ML(pointintneg,VectL2neg,L1neg)
    return xi,yi,zi,xr,yr,zr,xin,yin,zin,xrn,yrn,zrn, pointint, pointintneg, VectL2, VectL2neg

''' SetRange3dIndCorrecting (SR3IC): simply changing requirements. same as SR3I, just ignoring z.'''
def SR3IC (xrange, X,Y,Z, origin, sign): #given range, one point, origin, if it lies in or not
    if sign == 'pos':
        if (X-origin[0])**2 + (Z)**2 < xrange**2 and Y > 0:
            return True
        return False
    if sign == 'neg':
        if (X-origin[0])**2 + (Z)**2 < xrange**2 and Y < 0:
            return True
        return False
    if sign != 'pos' and sign != 'neg':
        print ('Error')

'''reflect_specellipsePOINTCORRECTING (RSEPC): Give coeffients of ellipse, points of rays, vectors of rays, radius of target, origin of target, and sign. Returns initial rays, reflecting rays, intersection points, and reflecting vectors. '''
def RSEPC(coeffellipse,pli,vectors, xrange, ellipseorigin, sign):
    xi = []
    yi = []
    zi = []
    xr = []
    yr = []
    zr = []
    vect = []
    pointints = []
    for i in range (0, len(vectors)):
        Vi = vectors[i]
        Pli = pli[i] #(or pli/original points of lines)
        xiL,yiL,ziL,xrL,yrL,zrL,xinL,yinL,zinL,xrnL,yrnL,zrnL, pointint, pointintneg, vectL2, vectL2neg = REPC(coeffellipse,Pli,Vi, ellipseorigin, sign)
        if SR3IC(xrange, pointint[0], pointint[1], pointint[2],ellipseorigin, sign) == True: 
            xi.append(xiL)
            yi.append(yiL)
            zi.append(ziL)
            xr.append(xrL)
            yr.append(yrL)
            zr.append(zrL)
            pointints.append(pointint) 
            vect.append(vectL2)
        if SR3IC(xrange, pointintneg[0], pointintneg[1], pointintneg[2], ellipseorigin, sign) == True:
            xi.append(xinL)
            yi.append(yinL)
            zi.append(zinL)
            xr.append(xrnL)
            yr.append(yrnL)
            zr.append(zrnL)
            pointints.append(pointintneg)
            vect.append(vectL2neg)
    return xi,yi,zi,xr,yr,zr, pointints, vect


def negvect(vect):
    vectset = [[-y for y in x] for x in vect]
    return vectset


