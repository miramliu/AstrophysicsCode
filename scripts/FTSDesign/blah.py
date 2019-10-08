def reflect(c,r,v,p,L):
        coeff1 = numpy.array(c) #plane coefficients
        VectL = numpy.array(v) #incident vector#defining points of incident vector
        VectLNorm = norm(v) #incident unit vector
        dU, N = plane_info(c,r) #gradient and normal of plane
        #reflected ray
        VectL2 = VectLNorm + 2*N #reflected vector
        VectLNorm2 = norm(VectL2) #reflected unit vector
        xp,yp,zp = make_plane(c,r) #plane
        pointint = intersec_point(c,v,p) #array and points of intersection
        xi,yi,zi = make_line(pointint,v,L) #incident line from intersection point
        xr,yr,zr = make_line(pointint,VectL2,L)
        return xi,yi,zi,xr,yr,zr,xp,yp,zp